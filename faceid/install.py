"""Install / uninstall the system integration (PAM, SELinux, wrappers).

Design rules, because this code edits authentication config:

  1. NEVER `required`.  Our line is always `auth sufficient`, so a failure
     can only ever fall through to the normal password prompt.  There is no
     configuration written by this module that can lock you out of sudo.
  2. Every file we touch is backed up to `<file>.faceid-bak` before the first
     change, and `faceid pam uninstall` restores them byte for byte.
  3. We refuse to write a PAM file that would end up with no non-faceid
     `auth` fallback line in it.
  4. Writes are atomic (temp file + rename) and the result is re-read and
     checked before we report success.
  5. `core.disabled = true` in the config is a one-line kill switch that
     needs no PAM edit at all.
"""

from __future__ import annotations

import os
import pwd
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

INSTALL_LIB = Path("/usr/local/lib/faceid")
BIN_DIR = Path("/usr/local/bin")
BIN = BIN_DIR / "faceid"
BIN_PAM = BIN_DIR / "faceid-pam"
MODEL_DIR = Path("/usr/share/faceid/models")
DATA_DIR = Path("/var/lib/faceid")
CONFIG_DIR = Path("/etc/faceid")
CONFIG_FILE = CONFIG_DIR / "config.toml"
PAM_DIR = Path("/etc/pam.d")
PAM_SO_CANDIDATES = [Path("/usr/lib64/security/pam_exec.so"),
                     Path("/usr/lib/security/pam_exec.so"),
                     Path("/lib/security/pam_exec.so"),
                     Path("/lib64/security/pam_exec.so")]
HOWDY_LINE_RE = re.compile(r"^\s*auth\s+.*pam_howdy\.so")
BAK_SUFFIX = ".faceid-bak"

DEFAULT_SERVICES = ["sudo", "sudo-i"]
OPTIONAL_SERVICES = ["polkit-1", "gdm-password", "sddm", "sddm-greeter",
                     "kde", "kscreenlocker", "su", "su-l", "login"]

PAM_HELPER_SH = """#!/bin/sh
# faceid PAM helper - invoked by pam_exec.so from the PAM stack.
# A hard timeout means a wedged camera can never block sudo or the greeter:
# on timeout we exit non-zero and PAM falls through to the password prompt.
exec /usr/bin/timeout -k 2 12 {bindir}/faceid auth "$@"
"""


def _helper_body(source_root: Path | None = None) -> str:
    """Prefer scripts/faceid-pam so the shipped helper cannot drift from the
    one that is installed; fall back to the inline template."""
    if source_root is not None:
        cand = Path(source_root) / "scripts" / "faceid-pam"
        try:
            if cand.is_file():
                return cand.read_text()
        except Exception:
            pass
    return PAM_HELPER_SH.format(bindir=BIN_DIR)

BIN_SH = """#!/bin/sh
# faceid - AI face authentication (installed copy)
export PYTHONPATH="{lib}:${{PYTHONPATH}}"
exec /usr/bin/python3 -m faceid.cli "$@"
"""


def sh(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, text=True,
                          capture_output=capture)


def pam_exec_path() -> Path | None:
    for p in PAM_SO_CANDIDATES:
        if p.is_file():
            return p
    return None


def _pam_line(exec_path: Path | None = None) -> str:
    ex = exec_path or pam_exec_path() or Path("pam_exec.so")
    name = ex.name if ex.parent.name != "" else "pam_exec.so"
    # `quiet` keeps syslog tidy; the helper writes its own notice to the tty.
    return f"auth       sufficient   {name} quiet {BIN_PAM}"


# --------------------------------------------------------------------------
def _read(path: Path) -> str:
    return path.read_text() if path.is_file() else ""


def _atomic_write(path: Path, text: str, mode: int = 0o644) -> None:
    tmp = Path(tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))[1])
    tmp.write_text(text)
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def _backup(path: Path) -> bool:
    bak = Path(str(path) + BAK_SUFFIX)
    if bak.exists():
        return False
    shutil.copy2(path, bak)
    return True


def _has_other_auth(text: str, exec_line: str) -> bool:
    """Is there still a password/other auth path in this stack?"""
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("@"):
            continue
        if not s.startswith("auth"):
            continue
        if "pam_exec.so" in s and str(BIN_PAM) in s:
            continue
        if HOWDY_LINE_RE.match(s):
            continue
        return True
    return False


def patch_service(service: str, exec_line: str, replace_howdy: bool = False,
                  dry_run: bool = False) -> dict:
    path = PAM_DIR / service
    out = {"service": service, "path": str(path), "changed": False,
           "skipped": None, "removed_howdy": 0, "backup": None}
    if not path.is_file():
        out["skipped"] = "no such PAM service"
        return out
    text = _read(path)
    if str(BIN_PAM) in text and "pam_exec.so" in text:
        out["skipped"] = "already installed"
        return out

    lines = text.splitlines()
    kept: list[str] = []
    removed = 0
    for line in lines:
        if replace_howdy and HOWDY_LINE_RE.match(line):
            removed += 1
            kept.append("# " + line + "   # disabled by faceid "
                        + time.strftime("%Y-%m-%d"))
            continue
        kept.append(line)
    if removed:
        out["removed_howdy"] = removed

    # Where to put our line: BEFORE the first "real" auth line, because PAM
    # evaluates in order and `auth include system-auth` / password-auth /
    # common-auth is itself `sufficient` - anything after it would never run.
    # Lines that must stay in front (SELinux permit, nologin, env) are kept.
    keep_first = ("pam_selinux_permit.so", "pam_nologin.so", "pam_env.so",
                  "pam_loginuid.so")
    insert_at = 0
    for i, line in enumerate(kept):
        s = line.strip()
        if not s or s.startswith("#"):
            insert_at = i + 1
            continue
        if s.startswith("@"):
            break                      # an @include may hold the password prompt
        if s.startswith("auth"):
            if any(k in s for k in keep_first):
                insert_at = i + 1
                continue
            break                      # first real auth line - go in front of it
        insert_at = i + 1              # some other module type: leave it before us
    new_lines = kept[:insert_at] + [exec_line] + kept[insert_at:]
    new_text = "\n".join(new_lines).rstrip() + "\n"

    if not _has_other_auth(new_text, exec_line):
        out["skipped"] = ("refusing to write: no password/other auth line would "
                          "remain (this would risk a lockout)")
        return out
    if dry_run:
        out["changed"] = True
        out["preview"] = new_text
        return out

    _backup(path)
    out["backup"] = str(path) + BAK_SUFFIX
    _atomic_write(path, new_text, mode=0o644)
    # read back and verify
    check = _read(path)
    if exec_line.strip() not in check or not _has_other_auth(check, exec_line):
        shutil.copy2(str(path) + BAK_SUFFIX, path)
        out["skipped"] = "verification failed after write - rolled back"
        return out
    out["changed"] = True
    return out


def unpatch_service(service: str, dry_run: bool = False) -> dict:
    path = PAM_DIR / service
    bak = Path(str(path) + BAK_SUFFIX)
    out = {"service": service, "restored": False, "removed_lines": 0}
    if not path.is_file():
        out["skipped"] = "no such PAM service"
        return out
    if bak.is_file():
        if not dry_run:
            shutil.copy2(bak, path)
            bak.unlink()
        out["restored"] = True
        return out
    text = _read(path)
    new = [l for l in text.splitlines()
           if not ("pam_exec.so" in l and str(BIN_PAM) in l)]
    removed = len(text.splitlines()) - len(new)
    out["removed_lines"] = removed
    if removed and not dry_run:
        _atomic_write(path, "\n".join(new).rstrip() + "\n")
    out["restored"] = bool(removed)
    return out


def resolve_services(services: list[str] | None) -> list[str]:
    if not services:
        return list(DEFAULT_SERVICES)
    if "all" in services:
        out = list(DEFAULT_SERVICES)
        for s in OPTIONAL_SERVICES:
            if (PAM_DIR / s).is_file():
                out.append(s)
        return list(dict.fromkeys(out))
    return list(dict.fromkeys(services))


def install_pam(services: list[str] | None = None, user: str | None = None,
                replace_howdy: bool = False, dry_run: bool = False) -> dict:
    log: list[str] = []
    if os.geteuid() != 0 and not dry_run:
        return {"ok": False, "log": ["must run as root (use: sudo faceid pam install)"]}
    ex = pam_exec_path()
    if ex is None and not dry_run:
        return {"ok": False,
                "log": ["pam_exec.so not found - is the pam package installed?"]}
    line = _pam_line(ex)
    svcs = resolve_services(services)
    log.append(f"pam line: {line}")
    log.append(f"services: {', '.join(svcs)}")
    results = []
    for s in svcs:
        r = patch_service(s, line, replace_howdy=replace_howdy, dry_run=dry_run)
        results.append(r)
        state = ("INSTALLED" if r["changed"] else
                 f"skipped ({r['skipped']})" if r["skipped"] else "unchanged")
        extra = f" [disabled {r['removed_howdy']} pam_howdy line(s)]" if r.get("removed_howdy") else ""
        log.append(f"  {s:18s} {state}{extra}")
        if dry_run and r.get("preview"):
            log.append("    --- preview ---")
            log.extend("    " + l for l in r["preview"].splitlines())
            log.append("    ----------------")
    if not dry_run:
        ensure_data_dir(user)
    ok = any(r["changed"] for r in results) or all(r["skipped"] == "already installed"
                                                  for r in results)
    log.append("")
    log.append("A `sufficient` line means the password prompt remains as fallback.")
    log.append("Keep a root shell open the first time you test, and verify with:")
    log.append("    faceid pam status")
    return {"ok": ok, "log": log, "results": results}


def uninstall_pam(services: list[str] | None = None, dry_run: bool = False) -> dict:
    log: list[str] = []
    if os.geteuid() != 0 and not dry_run:
        return {"ok": False, "log": ["must run as root"]}
    svcs = resolve_services(services) or list(DEFAULT_SERVICES)
    # always try everything we might have touched
    svcs = list(dict.fromkeys(svcs + DEFAULT_SERVICES + OPTIONAL_SERVICES))
    results = []
    for s in svcs:
        r = unpatch_service(s, dry_run=dry_run)
        results.append(r)
        if r.get("restored"):
            log.append(f"  {s:18s} restored"
                       + (f" ({r['removed_lines']} line(s) removed)"
                          if r.get("removed_lines") else " from backup"))
    log.append("all faceid PAM entries removed"
               if results else "nothing to do")
    return {"ok": True, "log": log, "results": results}


def pam_status() -> dict:
    line_re = re.compile(r"pam_exec\.so.*faceid-pam")
    out = {"installed": [], "files": {}, "howdy_present": [], "pam_exec": None}
    ex = pam_exec_path()
    out["pam_exec"] = str(ex) if ex else None
    for p in sorted(PAM_DIR.iterdir()):
        if not p.is_file() or p.name.endswith(BAK_SUFFIX):
            continue
        try:
            text = p.read_text()
        except Exception:
            continue
        if line_re.search(text):
            out["installed"].append(p.name)
            out["files"][p.name] = [l for l in text.splitlines() if line_re.search(l)]
        if "pam_howdy.so" in text:
            out["howdy_present"].append(p.name)
    out["backups"] = sorted(f.name for f in PAM_DIR.glob("*" + BAK_SUFFIX))
    return out


# --------------------------------------------------------------------------
def install_selinux_context() -> dict:
    if os.geteuid() != 0:
        return {"ok": False, "message": "must run as root"}
    msgs = []
    ok = True
    specs = [
        (BIN_DIR / "faceid", "bin_t"),
        (BIN_PAM, "bin_t"),
        (INSTALL_LIB, "lib_t"),
        (MODEL_DIR, "usr_t"),
    ]
    have_semanage = shutil.which("semanage") is not None
    for path, ctx in specs:
        if not path.exists():
            continue
        if have_semanage:
            r = subprocess.run(["semanage", "fcontext", "-a", "-t", ctx,
                                f"{path}(/.*)?"], capture_output=True, text=True)
            if r.returncode != 0 and "already defined" not in (r.stderr or ""):
                msgs.append(f"semanage failed for {path}: {r.stderr.strip()}")
                ok = False
    if shutil.which("restorecon"):
        subprocess.run(["restorecon", "-R", "-v", str(INSTALL_LIB)], capture_output=True)
        for p in (BIN, BIN_PAM):
            if p.exists():
                subprocess.run(["restorecon", "-v", str(p)], capture_output=True)
        msgs.append("restorecon applied")
    else:
        msgs.append("restorecon not found - install policycoreutils")
        ok = False
    return {"ok": ok, "message": "; ".join(msgs) or "nothing to do"}


def selinux_report() -> dict:
    out = {"enabled": False, "mode": "n/a", "denials": [], "count": 0}
    try:
        with open("/sys/fs/selinux/enforce") as fh:
            out["enabled"] = True
            out["mode"] = "enforcing" if fh.read().strip() == "1" else "permissive"
    except Exception:
        return out
    if shutil.which("ausearch"):
        r = subprocess.run(["ausearch", "-m", "avc", "-ts", "recent"],
                           capture_output=True, text=True)
        for block in r.stdout.split("----"):
            if "denied" not in block:
                continue
            if any(k in block for k in ("faceid", "pam_exec", f"tclass=chr_file",
                                        "video")):
                out["denials"].append(" ".join(block.split())[:400])
        out["count"] = len(out["denials"])
    return out


def selinux_policy_from_denials(name: str = "faceid_faceauth") -> dict:
    if os.geteuid() != 0:
        return {"ok": False, "message": "must run as root"}
    if not (shutil.which("ausearch") and shutil.which("audit2allow")):
        return {"ok": False, "message": "ausearch/audit2allow not available"}
    tmp = Path(tempfile.mkdtemp(prefix="faceid-selinux-"))
    try:
        r = subprocess.run(["ausearch", "-m", "avc", "-ts", "recent"],
                           capture_output=True, text=True)
        if "denied" not in r.stdout:
            return {"ok": True, "message": "no recent AVC denials recorded"}
        pp = tmp / name
        p = subprocess.run(["audit2allow", "-M", name], input=r.stdout,
                           capture_output=True, text=True, cwd=str(tmp))
        if p.returncode != 0:
            return {"ok": False, "message": p.stderr.strip() or "audit2allow failed"}
        pp_file = tmp / f"{name}.pp"
        if not pp_file.is_file():
            return {"ok": False, "message": "no policy module produced",
                    "te": (tmp / f"{name}.te").read_text()[:2000]}
        inst = subprocess.run(["semodule", "-i", str(pp_file)],
                              capture_output=True, text=True)
        if inst.returncode != 0:
            return {"ok": False, "message": inst.stderr.strip()}
        return {"ok": True, "message": f"installed policy module {name}",
                "te": (tmp / f"{name}.te").read_text()[:2000]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
def ensure_data_dir(user: str | None = None) -> dict:
    """The store must be writable by the human and readable by root (PAM)."""
    if os.geteuid() != 0:
        return {"ok": False, "message": "must run as root"}
    user = user or os.environ.get("SUDO_USER") or "root"
    try:
        pw = pwd.getpwnam(user)
        uid, gid = pw.pw_uid, pw.pw_gid
    except KeyError:
        uid, gid = 0, 0
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "gallery").mkdir(exist_ok=True)
    for p in (DATA_DIR, DATA_DIR / "gallery"):
        os.chown(p, uid, gid)
        os.chmod(p, 0o700)
    return {"ok": True, "message": f"{DATA_DIR} owned by {user} mode 0700 "
                                   f"(root can read it for PAM)"}


def _sync_tree(src: Path, dst: Path) -> int:
    n = 0
    for item in src.rglob("*"):
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in item.parts):
            continue
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if item.suffix in {".pyc", ".part"}:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        n += 1
    return n


def install_all(user: str | None = None, services: list[str] | None = None,
                replace_howdy: bool = False, dry_run: bool = False,
                source_root: Path | None = None) -> dict:
    log: list[str] = []
    if os.geteuid() != 0 and not dry_run:
        return {"ok": False, "log": ["run with sudo"]}
    src = Path(source_root or Path(__file__).resolve().parent.parent)
    user = user or os.environ.get("SUDO_USER") or "root"

    if dry_run:
        log.append(f"[dry-run] would install {src} -> {INSTALL_LIB}")
    else:
        INSTALL_LIB.mkdir(parents=True, exist_ok=True)
        n = _sync_tree(src / "faceid", INSTALL_LIB / "faceid")
        log.append(f"installed {n} source files -> {INSTALL_LIB}")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        m = 0
        for f in (src / "models").glob("*.onnx"):
            shutil.copy2(f, MODEL_DIR / f.name)
            m += 1
        log.append(f"installed {m} model file(s) -> {MODEL_DIR}")

        for p, body in ((BIN, BIN_SH.format(lib=INSTALL_LIB)),
                        (BIN_PAM, _helper_body(src))):
            _atomic_write(p, body, mode=0o755)
            log.append(f"wrote {p}")

        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        if not CONFIG_FILE.is_file():
            _atomic_write(CONFIG_FILE, DEFAULT_CONFIG_TEXT, mode=0o644)
            log.append(f"wrote {CONFIG_FILE}")
        else:
            log.append(f"kept existing {CONFIG_FILE}")

        r = ensure_data_dir(user)
        log.append(r["message"])

    pamres = install_pam(services=services, user=user, replace_howdy=replace_howdy,
                         dry_run=dry_run)
    log.extend(pamres["log"])

    if not dry_run:
        sr = install_selinux_context()
        log.append(f"selinux: {sr['message']}")
        rep = selinux_report()
        if rep.get("count"):
            log.append(f"selinux: {rep['count']} AVC denial(s) - run `sudo faceid "
                       f"selinux --build-policy` if face auth misbehaves")
    return {"ok": True, "log": log}


DEFAULT_CONFIG_TEXT = """# faceid configuration - see `faceid config` for every key and its default.
# This file only needs the keys you actually want to change.

[core]
# Set true to disable face auth instantly without touching /etc/pam.d.
disabled = false
abort_if_ssh = true
min_votes = 3
timeout = 5.0

[video]
device = "auto"
# Turn on after configuring the Windows-Hello style IR emitter
# (see `faceid ir`) - it makes replay attacks and dark rooms a non-issue.
prefer_ir = false

[match]
# `faceid calibrate --apply` will tighten this for you once you have
# enrolled. 0.363 is the published SFace reference point.
cosine_threshold = 0.40

[accessories]
enabled = true

[pam]
allowed_users = ["*"]
show_notice = true
"""


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="faceid-install")
    p.add_argument("--services", action="append", default=None,
                   help="extra PAM services (use 'all' for the full set)")
    p.add_argument("--user", default=None)
    p.add_argument("--replace-howdy", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    res = install_all(user=a.user, services=a.services,
                      replace_howdy=a.replace_howdy, dry_run=a.dry_run)
    for line in res["log"]:
        print(line)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
