"""Command line interface: `faceid <command>`.

The command that matters for sudo is `faceid auth`, which is what
pam_exec.so invokes.  Everything else is for enrolling, diagnosing and
tuning.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import __version__
from .config import load_config, paths as resolve_paths

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_ERR = 2


def _p(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _engine(cfg=None, paths=None):
    from .recognizer import Engine
    return Engine(cfg if cfg is not None else load_config(),
                  paths if paths is not None else resolve_paths())


def _progress_renderer(label: str, enabled: bool):
    state = {"last": 0.0}

    def render(ev: dict) -> None:
        now = time.time()
        if not enabled or now - state["last"] < 0.0:
            return
        if ev.get("face"):
            acc = ev.get("accessories", {}).get("claimed") or []
            extra = f" accessories={','.join(acc)}" if acc else ""
            if "kept" in ev:
                sys.stderr.write(f"\r{label}: frame {ev['frame']} kept "
                                 f"{ev['kept']}/{ev['of']}{extra}      ")
            else:
                sys.stderr.write(f"\r{label}: frame {ev['frame']} face "
                                 f"score={ev.get('score')}{extra}      ")
        elif ev.get("note"):
            sys.stderr.write(f"\r{label}: {ev['note']}                ")
        sys.stderr.flush()
    return render


# --------------------------------------------------------------------------
def cmd_doctor(args) -> int:
    eng = _engine()
    info = eng.probe_environment()
    info["version"] = __version__
    info["python"] = sys.version.split()[0]
    import cv2
    info["opencv"] = cv2.__version__
    info["uid"] = os.getuid()
    try:
        import grp
        gids = os.getgroups()
        info["group_names"] = sorted(grp.getgrgid(g).gr_name for g in gids if _g_exists(g))
        vid = grp.getgrnam("video").gr_gid
        info["in_video_group"] = vid in gids
    except Exception:
        info["in_video_group"] = None
    from .pam import pam_env, is_remote
    env = pam_env()
    info["pam_env"] = {k: v for k, v in env.items() if v}
    info["remote"], info["remote_reason"] = is_remote(env)
    info["selinux"] = _selinux_status()
    if args.json:
        _p(info)
    else:
        print(f"faceid {info['version']}  (python {info['python']}, opencv {info['opencv']})")
        print(f"  models        : {'OK' if info['models_ready'] else 'MISSING'}  {info['models']}")
        print(f"  data dir      : {info['data_dir']}")
        print(f"  accessory AI  : {'trained' if info['accessory_model'] else 'not trained (heuristic only)'}")
        print("  cameras:")
        for d in info["devices"]:
            if d.get("ir_hint"):
                kind = "mono/IR"
            else:
                kind = "colour"
            state = "usable" if d["usable"] else f"unusable ({d.get('error') or 'no frame'})"
            print(f"    {d['path']:14s} {kind:8s} {d['size']:>9s} "
                  f"bright={d.get('brightness')} contrast={d.get('contrast')}  {state}")
        ch = info["chosen_device"]
        print(f"  chosen camera : {ch['path'] if ch else 'NONE'} "
              f"({'mono/IR' if ch and ch.get('ir_hint') else 'colour'})")
        g = info["gallery"]
        print(f"  enrolments    : {g['total_embeddings']} embeddings")
        for u, d in g["users"].items():
            tags = ", ".join(f"{t}({i.get('count', 0)})" for t, i in d["tags"].items())
            print(f"    {u}: {tags or 'none'}")
        if g.get("calibration"):
            print(f"  calibration   : {g['calibration']}")
        print(f"  selinux       : {info['selinux']['mode']}")
        if info["selinux"].get("recent_denials"):
            print(f"                  {info['selinux']['recent_denials']} recent AVC "
                  f"denial(s) - see `faceid selinux`")
    return EXIT_OK


def _g_exists(gid: int) -> bool:
    try:
        import grp
        grp.getgrgid(gid)
        return True
    except Exception:
        return False


def _selinux_status() -> dict:
    out = {"enabled": False, "mode": "n/a", "recent_denials": 0}
    try:
        with open("/sys/fs/selinux/enforce") as fh:
            out["enabled"] = True
            out["mode"] = "enforcing" if fh.read().strip() == "1" else "permissive"
    except Exception:
        return out
    try:
        import subprocess
        r = subprocess.run(["ausearch", "-m", "avc", "-ts", "recent", "-c", "faceid"],
                           capture_output=True, text=True, timeout=10)
        out["recent_denials"] = r.stdout.count("avc:  denied")
    except Exception:
        pass
    return out


def cmd_devices(args) -> int:
    from .capture import probe_devices, choose_device
    cfg = load_config()
    devs = probe_devices(cfg.sec("video"))
    chosen = choose_device(cfg.sec("video"), devs)
    if args.json:
        _p({"devices": [d.as_dict() for d in devs],
            "chosen": chosen.as_dict() if chosen else None})
    else:
        for d in devs:
            print(f"{d.path:14s} mono={str(d.mono):5s} usable={str(d.usable):5s} "
                  f"{d.width}x{d.height} bright={d.brightness:6.1f} "
                  f"contrast={d.contrast:6.1f} {d.error}")
        print(f"chosen: {chosen.path if chosen else 'none'}")
    return EXIT_OK


def cmd_models(args) -> int:
    from .models import ensure_models, verify_models
    paths = resolve_paths()
    if args.download:
        res = ensure_models(paths.model_dir, force=args.force)
    else:
        res = verify_models(paths.model_dir)
    if args.json:
        _p(res)
    else:
        for name, r in res.items():
            print(f"{name:44s} {'OK' if r.get('ok') else 'PROBLEM'}  {r}")
    return EXIT_OK if all(r.get("ok") for r in res.values()) else EXIT_FAIL


def cmd_enroll(args) -> int:
    eng = _engine()
    user = args.user or os.environ.get("SUDO_USER") or _current_user()
    tags = args.tag or ["default"]
    out = []
    for tag in tags:
        print(f"Enrolling '{user}' with tag '{tag}' for {args.seconds:.0f}s - "
              f"look at the camera and turn your head slowly.", file=sys.stderr)
        rep = eng.enroll(user, tag=tag, seconds=args.seconds,
                         min_samples=args.min_samples,
                         on_frame=_progress_renderer("enroll", not args.quiet))
        if not args.quiet:
            sys.stderr.write("\r" + " " * 78 + "\r")
        print(f"  tag '{tag}': {'OK' if rep['ok'] else 'FAILED'} - {rep['reason']}",
              file=sys.stderr)
        if rep.get("predicted"):
            print(f"    accessory prediction: {rep['predicted']}", file=sys.stderr)
        out.append(rep)
    if len(out) > 1 and any(r["ok"] for r in out):
        tr = eng.train_accessories(user)
        print(f"  accessory model: {tr['reason']}", file=sys.stderr)
    if args.json:
        _p(out)
    return EXIT_OK if all(r["ok"] for r in out) else EXIT_FAIL


def cmd_train(args) -> int:
    eng = _engine()
    res = eng.train_accessories(args.user)
    if args.json:
        _p(res)
    else:
        print(res["reason"])
        for cls, info in (res.get("classes") or {}).items():
            print(f"  {cls:10s} {info}")
    return EXIT_OK if res.get("ok") else EXIT_FAIL


def cmd_verify(args) -> int:
    eng = _engine()
    users = args.user if isinstance(args.user, list) else ([args.user] if args.user else None)
    if not args.quiet:
        print("Hold still and look at the camera...", file=sys.stderr)
    res = eng.verify(users=users, seconds=args.seconds, explain=args.explain,
                     on_frame=_progress_renderer("verify", args.verbose))
    if not args.quiet:
        sys.stderr.write("\r" + " " * 78 + "\r")
    if args.json:
        _p(res.as_dict())
    else:
        print(f"{'VERIFIED' if res.ok else 'NO MATCH'}: {res.reason}")
        if args.explain and res.candidates:
            print("  candidates:")
            for c in res.candidates:
                print(f"    {c['user']:12s} tag={c['tag']:16s} score={c['score']:.4f} "
                      f"votes={c['votes']:3d} agree={c['accessory_agreement']:.2f} "
                      f"final={c['final']:.4f}")
    return EXIT_OK if res.ok else EXIT_FAIL


def cmd_test(args) -> int:
    eng = _engine()
    if not args.quiet:
        print("Identifying (no threshold applied)...", file=sys.stderr)
    res = eng.identify(seconds=args.seconds)
    if args.json:
        _p(res.as_dict())
    else:
        print(f"best guess : {res.user} (tag {res.tag}) score={res.score:.4f} "
              f"votes={res.votes}/{res.samples}")
        print(f"accessories: {res.accessories}")
        print(f"liveness   : {res.liveness}")
        print(f"decision   : {'ACCEPT' if res.ok else 'REJECT'} - {res.reason}")
        if args.explain and res.candidates:
            print("  all tag scores:")
            for c in sorted(res.candidates, key=lambda x: -x["final"]):
                print(f"    {c['user']:12s} tag={c['tag']:16s} score={c['score']:.4f} "
                      f"votes={c['votes']:3d} agree={c['accessory_agreement']:.2f} "
                      f"bonus={c['bonus']:+.4f} final={c['final']:.4f}")
    return EXIT_OK if res.ok else EXIT_FAIL


def cmd_calibrate(args) -> int:
    eng = _engine()
    res = eng.calibrate(user=args.user, seconds=args.seconds, attempt=args.attempt)
    if args.json:
        _p(res)
    else:
        _p(res)
    if not res.get("ok"):
        return EXIT_FAIL
    if args.apply:
        cal = eng.gallery.meta.setdefault("calibration", {})
        s = float(res["score"])
        new_thr = round(max(0.20, s - args.margin), 4)
        eng.gallery.set_calibration(cosine_threshold=new_thr,
                                    measured_score=round(s, 4),
                                    measured_tag=res.get("tag"))
        eng.gallery.save()
        print(f"applied: cosine_threshold = {new_thr} (measured genuine {s:.4f} "
              f"minus margin {args.margin})")
    return EXIT_OK


def cmd_list(args) -> int:
    eng = _engine()
    s = eng.gallery.summary()
    if args.json:
        _p(s)
    else:
        print(f"store: {s['data_dir']}   models: {s['model_dir']}")
        print(f"total: {s['total_embeddings']} embeddings")
        for u, d in s["users"].items():
            print(f"  {u}")
            for t, i in d["tags"].items():
                acc = ", ".join(i.get("accessories") or []) or "bare"
                print(f"    {t:20s} {i.get('count', 0):4d} frames  "
                      f"wearing: {acc:24s} updated {i.get('updated','')}")
        if s.get("calibration"):
            print(f"  calibration: {s['calibration']}")
    return EXIT_OK


def cmd_remove(args) -> int:
    eng = _engine()
    user = args.user or _current_user()
    n = eng.gallery.remove(user, args.tag)
    eng.gallery.save()
    print(f"removed {n} embeddings for {user}" + (f" tag {args.tag}" if args.tag else ""))
    return EXIT_OK


def cmd_ir(args) -> int:
    from .capture import probe_devices
    cfg = load_config()
    devs = probe_devices(cfg.sec("video")) if not args.no_probe else []
    ir = [d for d in devs if d.is_ir_hint]
    out = {"ir_devices": [d.as_dict() for d in ir],
           "all": [d.as_dict() for d in devs]}
    if args.json:
        _p(out)
    else:
        print("IR / monochrome sensors found:")
        for d in ir:
            print(f"  {d.path} {d.width}x{d.height} brightness={d.brightness:.1f} "
                  f"contrast={d.contrast:.1f}")
        if not ir:
            print("  none")
        print("\nHow to actually use IR (much stronger anti-spoofing and it works "
              "in the dark):")
        print("  1. install linux-enable-ir-emitter, then run:")
        print("       sudo linux-enable-ir-emitter configure")
        print("     pick the device whose picture becomes a visible greyscale face.")
        print("  2. re-run `faceid devices` - the IR node should now be bright.")
        print("  3. faceid config --set video.prefer_ir true")
        print("     (or `prefer_ir` in /etc/faceid/config.toml)")
        print("  A monitor emits no IR, so requiring IR defeats photo/video replay.")
    return EXIT_OK


def cmd_config(args) -> int:
    cfg = load_config()
    if args.set:
        key, _, val = args.set.partition("=")
        path = args.file or ("/etc/faceid/config.toml" if os.geteuid() == 0
                             else os.path.expanduser("~/.config/faceid/config.toml"))
        _write_config(path, key, val)
        print(f"wrote {key} = {val} to {path}")
        return EXIT_OK
    if args.json:
        _p(cfg)
    else:
        print(_toml_dump(cfg))
    return EXIT_OK


def _toml_dump(d: dict, prefix: str = "") -> str:
    lines = []
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"[{prefix}{k}]")
            lines.append(_toml_dump(v, prefix=f"{prefix}{k}."))
        else:
            lines.append(f"{k} = {json.dumps(v)}")
    return "\n".join(x for x in lines if x)


def _write_config(path: str, key: str, value: str) -> None:
    """Read-modify-write a TOML key without a TOML writer dependency."""
    import re
    from pathlib import Path
    section, _, name = key.rpartition(".")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = p.read_text() if p.is_file() else ""
    val = value
    low = value.lower()
    if low in ("true", "false"):
        val = low
    elif re.fullmatch(r"-?\d+", value):
        val = value
    elif re.fullmatch(r"-?\d*\.\d+", value):
        val = value
    else:
        val = json.dumps(value)
    lines = text.splitlines()
    out, in_sec, done = [], False, False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if in_sec and not done:
                out.append(f"{name} = {val}")
                done = True
            in_sec = stripped[1:-1].strip() == section
            out.append(line)
            continue
        if in_sec and re.match(rf"^{re.escape(name)}\s*=", stripped):
            out.append(f"{name} = {val}")
            done = True
            continue
        out.append(line)
    if not done:
        if section and not any(l.strip() == f"[{section}]" for l in out):
            if out and out[-1].strip():
                out.append("")
            out.append(f"[{section}]")
        out.append(f"{name} = {val}")
    p.write_text("\n".join(out).rstrip() + "\n")


def cmd_selinux(args) -> int:
    from .install import selinux_report, selinux_policy_from_denials
    rep = selinux_report()
    if args.build_policy:
        res = selinux_policy_from_denials()
        _p(res)
        return EXIT_OK
    _p(rep)
    return EXIT_OK


def cmd_pam(args) -> int:
    from .install import (install_pam, uninstall_pam, pam_status,
                          install_selinux_context)
    if args.action == "install":
        res = install_pam(services=args.services, user=args.user,
                          replace_howdy=args.replace_howdy, dry_run=args.dry_run)
        for line in res["log"]:
            print(line)
        if args.selinux and not args.dry_run:
            print(install_selinux_context()["message"])
        return EXIT_OK if res.get("ok") else EXIT_FAIL
    if args.action == "uninstall":
        res = uninstall_pam(services=args.services, dry_run=args.dry_run)
        for line in res["log"]:
            print(line)
        return EXIT_OK if res.get("ok") else EXIT_FAIL
    if args.action == "log":
        return _pam_log(clear=args.clear)
    _p(pam_status())
    return EXIT_OK


def _pam_log(clear: bool = False, lines: int = 120) -> int:
    """Show the PAM helper's diagnostic log.

    Without this, a failing `pam_exec.so quiet` line is invisible: to the user
    it just looks like sudo asked for a password.
    """
    from .config import paths as _paths
    log = _paths().data_dir / "pam-debug.log"
    if clear:
        try:
            log.write_text("")
            print(f"cleared {log}")
        except Exception as exc:
            print(f"could not clear {log}: {exc}", file=sys.stderr)
            return EXIT_ERR
        return EXIT_OK
    if not log.is_file():
        print(f"no PAM debug log yet at {log}")
        print("It is written by /usr/local/bin/faceid-pam on every attempt; "
              "run a `sudo` and try again.")
        return EXIT_OK
    lines_out = log.read_text(errors="replace").splitlines()
    print(f"--- {log} (last {min(lines, len(lines_out))} lines) ---")
    for line in lines_out[-lines:]:
        print(line)
    return EXIT_OK


def cmd_auth(args) -> int:
    """The entry point pam_exec.so calls.  Exit 0 = authenticated."""
    from .pam import authenticate, should_attempt, pam_env
    cfg = load_config()
    service = args.service or None
    dec = should_attempt(cfg, service)
    if not dec.attempt:
        if args.debug:
            print(f"faceid auth: skipped - {dec.reason}", file=sys.stderr)
        if args.why:
            print(f"skipped: {dec.reason}", file=sys.stderr)
        return EXIT_FAIL
    users = args.user or None
    res = authenticate(cfg, service=service, users=users, seconds=args.seconds,
                       explain=args.debug)
    if args.debug:
        _p(res.as_dict())
    if args.why:
        # One line, always - the PAM helper logs this, so a failed sudo is
        # never an unexplained "a password is required" ever again.
        print(f"{'ACCEPT' if res.ok else 'REJECT'} {res.reason}", file=sys.stderr)
    if res.ok:
        cfg_pam = cfg.sec("pam")
        if cfg_pam.get("show_notice", True):
            _notify_tty(f"  \033[32m✱\033[0m face recognised: {res.user}"
                        f"{'' if not res.tag or res.tag == 'default' else f' [{res.tag}]'}\n")
    return EXIT_OK if res.ok else EXIT_FAIL


def _notify_tty(msg: str) -> None:
    """PAM runs us with our stdout detached; talk to the real terminal."""
    for path in ("/dev/tty",):
        try:
            with open(path, "w") as fh:
                fh.write(msg)
                fh.flush()
            return
        except Exception:
            continue


def cmd_version(args) -> int:
    print(__version__)
    return EXIT_OK


def _current_user() -> str:
    try:
        import pwd
        return pwd.getpwuid(os.getuid()).pw_name
    except Exception:
        return os.environ.get("USER", "user")


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="machine readable output")

    p = argparse.ArgumentParser(
        prog="faceid",
        description="AI face authentication with hat/headphone awareness",
        parents=[common])
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("doctor", help="full environment report", parents=[common])
    sp.set_defaults(func=cmd_doctor)

    sp = sub.add_parser("devices", help="list camera devices", parents=[common])
    sp.set_defaults(func=cmd_devices)

    sp = sub.add_parser("models", help="check or download the AI models", parents=[common])
    sp.add_argument("--download", action="store_true")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_models)

    sp = sub.add_parser("enroll", help="enroll a face, optionally tagged with accessories",
                        parents=[common])
    sp.add_argument("--user", default=None)
    sp.add_argument("--tag", action="append",
                    help="what you are wearing: default | hat | headphones | "
                         "'hat,headphones' (repeatable)")
    sp.add_argument("--seconds", type=float, default=8.0)
    sp.add_argument("--min-samples", type=int, default=12)
    sp.add_argument("--quiet", action="store_true")
    sp.set_defaults(func=cmd_enroll)

    sp = sub.add_parser("train", help="train the accessory classifier from tagged enrolments",
                        parents=[common])
    sp.add_argument("--user", default=None)
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("verify", help="authenticate (exit 0 on success)", parents=[common])
    sp.add_argument("--user", default=None)
    sp.add_argument("--seconds", type=float, default=None)
    sp.add_argument("--explain", action="store_true", help="show all tag scores")
    sp.add_argument("--verbose", action="store_true", help="live per-frame progress")
    sp.add_argument("--quiet", action="store_true")
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("test", help="identify without applying the accept threshold",
                        parents=[common])
    sp.add_argument("--seconds", type=float, default=None)
    sp.add_argument("--explain", action="store_true")
    sp.add_argument("--quiet", action="store_true")
    sp.set_defaults(func=cmd_test)

    sp = sub.add_parser("calibrate", help="measure genuine scores and tune the threshold",
                        parents=[common])
    sp.add_argument("--user", default=None)
    sp.add_argument("--seconds", type=float, default=6.0)
    sp.add_argument("--attempt", type=int, default=0)
    sp.add_argument("--apply", action="store_true")
    sp.add_argument("--margin", type=float, default=0.08)
    sp.set_defaults(func=cmd_calibrate)

    sp = sub.add_parser("list", help="show enrolments", parents=[common])
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("remove", help="delete an enrolment", parents=[common])
    sp.add_argument("--user", default=None)
    sp.add_argument("--tag", default=None)
    sp.set_defaults(func=cmd_remove)

    sp = sub.add_parser("ir", help="IR camera info and how to enable the emitter",
                        parents=[common])
    sp.add_argument("--no-probe", action="store_true")
    sp.set_defaults(func=cmd_ir)

    sp = sub.add_parser("config", help="show or set configuration", parents=[common])
    sp.add_argument("--set", metavar="section.key=value", default=None)
    sp.add_argument("--file", default=None)
    sp.set_defaults(func=cmd_config)

    sp = sub.add_parser("selinux", help="SELinux denials and policy generation",
                        parents=[common])
    sp.add_argument("--build-policy", action="store_true")
    sp.set_defaults(func=cmd_selinux)

    sp = sub.add_parser("pam", help="install/uninstall the PAM integration",
                        parents=[common])
    sp.add_argument("action", choices=["install", "uninstall", "status", "log"])
    sp.add_argument("--services", action="append", default=None)
    sp.add_argument("--user", default=None)
    sp.add_argument("--replace-howdy", action="store_true",
                    help="remove existing pam_howdy lines")
    sp.add_argument("--selinux", action="store_true", help="also set SELinux file context")
    sp.add_argument("--clear", action="store_true", help="with `pam log`: truncate it")
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(func=cmd_pam)

    sp = sub.add_parser("auth", help="PAM helper entry point", parents=[common])
    sp.add_argument("--user", default=None)
    sp.add_argument("--service", default=None)
    sp.add_argument("--seconds", type=float, default=None)
    sp.add_argument("--debug", action="store_true")
    sp.add_argument("--why", action="store_true",
                    help="always print a single-line verdict (used by the PAM helper)")
    sp.add_argument("--notice", action="store_true", default=True)
    sp.set_defaults(func=cmd_auth)

    sp = sub.add_parser("version", help="print version", parents=[common])
    sp.set_defaults(func=cmd_version)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_ERR
    except Exception as exc:
        if getattr(args, "json", False):
            _p({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        else:
            print(f"faceid: {type(exc).__name__}: {exc}", file=sys.stderr)
        if os.environ.get("FACEID_DEBUG"):
            raise
        return EXIT_ERR


if __name__ == "__main__":
    sys.exit(main())
