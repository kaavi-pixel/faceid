# HANDOFF.md

Operational handoff for the `faceid` work on this machine.
Written 2026-11-02. Repo: <https://github.com/kaavi-pixel/faceid>
Working tree: `~/faceid` · commit `4fbacb5`

Read `overallplan.md` for where this is going. Read this to pick it up
without re-deriving anything.

---

## 1. TL;DR

The software is built, installed, tested (**34/34**) and published.

The **PAM integration is proven** — authenticating against the real `sudo`
stack returns success with no password:

```
PAM_USER=$USER : pam_authenticate -> 0 (Success)
PAM_USER=root  : pam_authenticate -> 0 (Success)
helper log     : rc=0 user=$USER ruser=$USER service=sudo
                 rc=0 user=root   ruser=      service=sudo
```

What is **not** yet proven is a bare `sudo` command end-to-end. The two
attempts so far returned `rc=1` from the face pipeline (no face in frame) and
the installed helper was too old to record *why*.

**The single next action:**

```bash
cd ~/faceid && sudo ./install.sh && sudo -k && sudo -n true; faceid pam log
```

Look at the camera when it runs. `sudo -n` cannot prompt, so success means the
face alone satisfied PAM. The log line carries the verdict either way.

---

## 2. Machine state, right now

| Thing | Value |
|---|---|
| Installed package | `/usr/local/lib/faceid` |
| CLI / helper | `/usr/local/bin/faceid`, `/usr/local/bin/faceid-pam` |
| Models | `/usr/share/faceid/models/*.onnx` (YuNet + SFace, sha256-verified) |
| Config | `/etc/faceid/config.toml` |
| Store | `/var/lib/faceid` — owned by `$USER`, mode 0700, readable by root |
| Enrolments | user **$USER**: `hat` (8 frames, headwear), `headphones` (21 frames, earwear) |
| Threshold | **not calibrated** — using the shipped `0.40` |
| Accessory model | trained on 29 frames across 2 tags |
| PAM | line present in `sudo` and `sudo-i`, **first**, before `pam_howdy` and `system-auth` |
| Howdy | still active and broken (see §6.4) |
| Desktop | GNOME. **No Plasma installed** — KDE paths untested |
| IR emitter | not installed; `/dev/video2` is a dark 600×600 mono sensor |
| Free disk | ~3.3 GB — do not vendor the 39 MB of models into git |

### Stale installed files `[!]`

As of this writing these differ from the working tree, which is why the sudo
test cannot yet explain itself:

```
/usr/local/lib/faceid/faceid/cli.py   STALE
/usr/local/bin/faceid-pam             STALE (no --why, so no reason is logged)
```

`pam.py`, `config.py`, `recognizer.py`, `install.py` are up to date.
Re-running `install.sh` is idempotent and fixes both.

### Re-check the state in one shot

```bash
cd ~/faceid
for f in pam.py config.py recognizer.py cli.py install.py; do
  diff -q faceid/$f /usr/local/lib/faceid/faceid/$f >/dev/null 2>&1 \
    && echo "$f ok" || echo "$f STALE"
done
grep -q -- '--why' /usr/local/bin/faceid-pam && echo "helper ok" || echo "helper STALE"
/usr/bin/python3 -c "
import sys; sys.path.insert(0,'$HOME/faceid')
from faceid.config import paths; from faceid.gallery import Gallery
g=Gallery(paths()); print('store:', paths().data_dir, g.users(), len(g.emb))
for u in g.users():
    for t in g.tags(u): print(' ', u, t, len(g.matrix(u,t)[0]))
print('calibration:', g.meta.get('calibration') or 'NOT SET')"
/usr/bin/python3 tests/test_faceid.py | tail -1
```

---

## 3. Verified vs unverified

| | Status |
|---|---|
| Detection on a real face, incl. cap and profile | **verified** 0.937 / 20-of-20 at 0.92 |
| Accessory detection (headwear, earwear) | **verified** — correctly said "no hat" when there was none |
| Tag routing picks the right gallery | **verified** — `hat` 0.7027 vs `headphones` 0.9865 |
| Impostor / single-frame / static-image rejection | **verified** in tests |
| PAM line ordering + password fallback intact | **verified** by reading `/etc/pam.d/sudo` |
| Real `sudo` stack authenticates without a password | **verified** via libpam harness |
| A bare `sudo` command succeeding | **not verified** — see §1 |
| GDM login | **not installed** |
| KDE Plasma (SDDM, lock screen) | **untested, no Plasma on this box** |
| polkit graphical prompt | **not exercised** |
| IR camera | **not configured** |

---

## 4. Debugging playbook

### `sudo` asks for a password

```bash
faceid pam log            # one line per attempt: rc, PAM_USER, PAM_RUSER, service
faceid pam log --clear    # start from a clean slate
```

Read it like this:

- **Log absent after a `sudo` attempt** → `pam_exec` never executed the helper.
  The problem is the wiring, not the face. Check the PAM file and SELinux
  (`sudo faceid selinux --build-policy`).
- **Log present with `rc=1`** → the helper ran; read the `REJECT <reason>` line
  after it. If it says no face was captured, the camera is not pointed at you.
- **`rc=127`** → a command in the helper was not found. The helper deliberately
  uses absolute paths; keep it that way.

### Test the real PAM stack without touching sudo

```bash
python3 tests/pam_harness.py --service sudo --user $USER
sudo python3 tests/pam_harness.py --service sudo --probe
```

Its conversation function refuses to answer prompts — the same semantics as
`sudo -n` — so a success means biometrics alone satisfied the stack.
`--probe` injects (and restores) an `auth optional pam_exec.so` marker line so
you can tell "pam_exec never ran" apart from "our helper ran and failed".

### Is the camera even looking at a face?

```bash
faceid test --explain           # best guess + every tag score, no threshold applied
faceid devices                  # what each /dev/videoN really is
```

A confident face scores **~0.9**. Boxes appearing at 0.07–0.16 are false
positives on scenery — that means nobody is in frame. This has wasted time
repeatedly on this machine: the camera often points at the ceiling.

### Emergency / rollback

```bash
sudo faceid pam uninstall       # restores every PAM file from <file>.faceid-bak
faceid config --set core.disabled=true    # kill switch, no PAM edit needed
```

Keep a root shell open the first time you change anything PAM-related. The
design makes lockout structurally impossible (`auth sufficient`, never
`required`), but do not rely on that as your only safety net.

---

## 5. File map

```
faceid/capture.py       device probing (UVC metadata nodes!), framing, quality gates
faceid/detect.py        YuNet detection, multi-scale with a fast path
faceid/embed.py         SFace embeddings, CLAHE + gamma variants averaged
faceid/headfeatures.py  64 head-region features for accessory awareness
faceid/accessories.py   logistic-regression classifier + geometric prior
faceid/gallery.py       tagged enrolment store (npz + json, atomic writes)
faceid/liveness.py      motion-band + consistency + optional moire checks
faceid/recognizer.py    the engine: capture -> score -> decide
faceid/pam.py           should we attempt, and as whom
faceid/install.py       PAM patching, SELinux contexts, wrappers
faceid/cli.py           all commands
tests/test_faceid.py    34 tests, standalone runner
tests/pam_harness.py    real PAM stack via libpam/ctypes
scripts/faceid-pam      the pam_exec helper (absolute paths, timeout, logging)
overallplan.md          roadmap        <- this repo
HANDOFF.md              this file
```

---

## 6. Traps already paid for — do not re-learn these

Each of these presented as a *different* bug than it was.

### 6.1 PAM line ordering
**Symptom:** face auth "not working", password always asked.
**Cause:** a line appended *after* `auth include system-auth` is unreachable —
that include is itself `sufficient`, so it prompts first and never yields.
**Fix:** insert *before* the first real auth line, after only
`pam_selinux_permit` / `pam_nologin` / `pam_env`.

### 6.2 Path-based PAM service names are ignored
**Symptom:** even `/bin/true` "fails" in a PAM stack you wrote yourself.
**Cause:** `pam_start("/path/to/file")` silently falls through to
`/etc/pam.d/other`, which is `auth required pam_deny.so`. Looks exactly like a
broken module.
**Fix:** always authenticate against a real service name. This one sent the
work down a false trail for a while — `pam_exec` was working the whole time.

### 6.3 CLI and PAM helper resolving different stores
**Symptom:** enrolments made in a terminal are invisible to sudo; PAM falls
back to the password and everything looks fine.
**Cause:** the unprivileged CLI defaulted to `~/.local/share/faceid`, the
root-run helper to `/var/lib/faceid` (which the installer had created empty).
**Fix:** `resolve_data_dir()` prefers the shared store whenever it is usable;
the home path is now a symlink to it; there is a test for this.

### 6.4 `sudo` does not name the human
**Symptom:** candidates resolve to `["root"]`, nothing matches.
**Cause:** `sudo` sets `PAM_USER=root` (the *target*). `PAM_RUSER` is set here
but is not something to depend on.
**Fix:** `pam.fallback_to_enrolled` (default on) widens candidates to the
machine's enrolled users when none of them owns a face. Turn it off on a
shared machine — it means any enrolled user can be matched.

### 6.5 `mean` over accessory probabilities
**Symptom:** enrolment auto-tagged `default` while clearly wearing
headphones, so tag routing silently stopped working while still appearing to
function.
**Cause:** earwear measured a steady 0.60 across near-identical frames, but a
`mean` across a 12 s enrolment where the head turned fell to 0.47 — under the
0.55 gate.
**Fix:** aggregate with a high percentile (`p75`), the *same* aggregation at
enrol and verify. Tell-tale: `verify --explain` shows `agree=0.00` for every
tag.

### 6.6 Enrolment starvation
**Symptom:** "only 1 usable frame (need 12)" for a user sitting still.
**Cause:** the capture loop skipped frames that resembled ones already kept.
**Fix:** collect everything to a hard cap, then greedily pick a diverse subset
with a top-up fallback so a motionless user can always enrol.

### 6.7 `pam_exec.so quiet` swallows everything
**Symptom:** "sudo still asks for my password" with zero diagnostics in
syslog.
**Fix:** the helper logs one line per attempt plus the child's `--why` verdict;
`faceid pam log` reads it. If you take one thing from this file: keep that.

### 6.8 Enrolling under the wrong username
**Symptom:** total silence — no match, ever.
**Cause:** the gallery is keyed by **Unix username**; enrolling with a
`--user` that is not the login name produces a gallery PAM will never ask for.
**Fix:** `faceid enroll` defaults to the current account — do not override it.
The failure message now lists the names the gallery actually holds.

---

## 7. Open questions for the operator

1. Should `faceid` also take over **GDM login** and, later, the **Plasma lock
   screen**? Both are higher risk than sudo.
2. Disable Howdy (`--replace-howdy`)? It is currently broken and adds its own
   timeout to every `sudo`.
3. Configure the **IR emitter**? Biggest remaining robustness win, and it
   makes replay attacks a non-issue.
4. Is this machine single-user? If not, revisit `fallback_to_enrolled` and
   `allowed_users`.
5. Worth packaging as an RPM/COPR, or keep it as a clone-and-install script?

---

## 8. Rebuild from scratch

```bash
git clone https://github.com/kaavi-pixel/faceid.git ~/faceid
cd ~/faceid
sudo dnf install python3-opencv python3-numpy      # usually already present
/usr/bin/python3 -m faceid.models_dl ./models      # ~39 MB, sha256 checked
export PYTHONPATH=~/faceid
/usr/bin/python3 tests/test_faceid.py              # expect 34/34
/usr/bin/python3 -m faceid.cli enroll --tag auto   # look at the camera
sudo ./install.sh                                  # wrappers, PAM, SELinux
faceid verify --explain
```

Note: use **`/usr/bin/python3`**. A virtualenv on this machine has no `cv2`,
and the system `python3-opencv` already has everything required.
