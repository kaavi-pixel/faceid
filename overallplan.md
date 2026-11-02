# overallplan.md — faceid roadmap

**Goal:** face authentication on Fedora that survives the things that actually
break the alternatives (hat, headphones, glasses, hood, profile, dim room),
that can approve `sudo`, and that **cannot lock anyone out**.

Reference implementation: <https://github.com/kaavi-pixel/faceid>
Installed on this machine at `/usr/local/lib/faceid`, store at `/var/lib/faceid`.

Legend: `[x]` verified with real output · `[~]` works but incomplete/untested
in the target environment · `[ ]` not started · `[!]` blocked

---

## Phase 0 — the build `[x]`

| Item | Status | Evidence |
|---|---|---|
| YuNet CNN detection replaces dlib HOG | `[x]` | 0.937 confidence on a real capped face; 20/20 frames at 0.92, including a profile view |
| SFace 128-d embeddings, no extra runtime dep | `[x]` | runs on OpenCV's DNN engine; only `python3-opencv` + `python3-numpy` needed |
| Tagged galleries (`default`/`hat`/`headphones`/…) | `[x]` | 2 tags live; routing demonstrated in `verify --explain` |
| Per-user learned accessory classifier | `[x]` | trained on 29 frames; `source: learned+heuristic` |
| Multi-frame voting, margin, liveness | `[x]` | impostor rejected; 1-frame attempt rejected; static image rejected |
| Safe PAM integration | `[x]` | `auth sufficient` only; backup/verify/rollback; 12 s hard timeout |
| Cannot lock out | `[x]` | password fallback proven intact after every patch |
| Test suite | `[x]` | **34/34 passing**, no pytest required |
| Published public repo | `[x]` | verified read-back, no biometric/binary blobs |
| PAM test harness (ctypes/libpam) | `[x]` | drives the real `sudo` stack without a compiler or root |

---

## Phase 1 — close the loop on sudo `[~]`

The PAM wiring is proven: authenticating against the **real** `sudo` stack
returns `pam_authenticate -> 0 (Success)` with no password, for both
`PAM_USER=$USER` and `PAM_USER=root`.

- [x] Identify why a bare `sudo` never used the face — `PAM_TTY=/dev/pts/N`
      was misread as a remote session, so the helper skipped itself every
      time (see trap 9). Fixed; the failing environment now returns
      `ACCEPT ... exit=0`.
- [x] Make failures self-explanatory — helper logs `--why`, `faceid pam log`
- [ ] Re-run `sudo ./install.sh` to push the fixed `pam.py` to `/usr/local`
- [ ] Run one `sudo` **while looking at the camera**; confirm zero keystrokes
- [ ] Read `faceid pam log`; if `rc=1`, act on the logged reason
- [ ] Confirm over several days that it is not flaky

**Exit criteria:** `sudo -k; sudo -n true` succeeds three times in a row with
the face in frame and fails cleanly when it is not.

---

## Phase 2 — enrolment completeness `[~]`

- [ ] **Enrol a `default` (bare face) tag.** The store currently has only
      `hat` and `headphones`; there is no bare-face gallery, so the routing
      has nothing to fall back to when you are wearing neither.
- [ ] Enrol the remaining states you actually appear in: `glasses`,
      `hat,headphones`, `hood`, and a dim-room capture.
- [ ] Re-run `faceid train` after each — the classifier is only as good as
      the states it has seen. With 2 tags it is close to trivial (2 classes,
      `train_accuracy 1.0` is memorisation); 4+ tags is where it earns its
      keep.
- [ ] `faceid calibrate --apply` to set the threshold from *measured* genuine
      scores instead of the shipped 0.40 default. Currently **not set**.
- [ ] Sanity-check the accessory threshold with a deliberate mismatch
      (enrol bare, verify hatted) and confirm the reason is informative.

---

## Phase 3 — integration breadth `[~]`

- [ ] **GDM login.** Our line is only in `sudo` and `sudo-i`; the existing
      `pam_howdy.so` line in `gdm-password` is untouched. Decide whether to
      install there, and note that GDM runs its PAM stack in a restricted
      context — camera access and SELinux need checking before trusting it.
- [ ] **KDE Plasma — untested, no Plasma on this machine.** `sddm`,
      `sddm-greeter` and `kscreenlocker` names are written from the Plasma
      layout but have never been executed. Do this in a VM.
      Note the extra requirement: Plasma's lock screen runs as the session
      user, so that user needs camera access (`video` group as fallback).
- [ ] **polkit (`polkit-1`)** for graphical admin prompts — written, not yet
      exercised.
- [ ] Decide on Howdy: it is currently **still active and broken** (prints
      "Failure, timeout reached" and adds its own delay to every `sudo`).
      Either `--replace-howdy` or leave both and accept the cost. Recommend
      disabling.
- [ ] Compare `system-auth` vs `sudo` placement once more after any change.

---

## Phase 4 — IR camera (largest remaining robustness win) `[ ]`

`/dev/video2` is a 600×600 monochrome IR sensor sitting at brightness ~7 with
the emitter off. Enabling it converts three weak areas into solved ones:

- works in a pitch-dark room;
- ignores a cap brim's shadow entirely;
- a monitor or a printed photo emits no IR, which defeats replay attacks
  properly rather than by heuristic.

- [ ] Install and run `linux-enable-ir-emitter configure`; pick the device
      whose preview becomes a visible greyscale face
- [ ] Re-check with `faceid devices`; the IR node should be bright
- [ ] `faceid config --set video.prefer_ir=true`
- [ ] Enrol an IR gallery (`--tag default` is fine; IR captures are their own
      condition) and measure detection/verification rates
- [ ] Document the fallback when the emitter is unavailable

---

## Phase 5 — polish and distribution `[ ]`

- [ ] Package as an RPM / COPR so this is not a `git clone && sudo ./install.sh`
- [ ] `docs/` with a screenshot of `verify --explain` and the tag-routing table
- [ ] A short ADR for the "tags over single template" decision — it is the
      whole design and the least obvious part
- [ ] Optional: systemd user unit to keep a warm engine, removing the ~0.6 s
      model load from the sudo critical path
- [ ] Optional: `faceid selinux --build-policy` in CI-style check after fresh
      installs on other machines

---

## Explicit non-goals / limits to state honestly

- **Not a certified anti-spoofing system.** The liveness gate is a speed
  bump. IR (Phase 4) is the real answer. Do not deploy this where face auth
  is the only factor protecting something valuable.
- **No encrypted-home support at the login screen.** If `/home` is not
  mounted when GDM authenticates, a home-store gallery is unreachable. The
  system store at `/var/lib/faceid` avoids this; that is why it is the default
  once installed.
- **Single-machine, single-operator design.** `pam.fallback_to_enrolled =
  true` is what makes sudo resolve, and it means any enrolled user can be
  matched. Turn it off on a shared machine.
- **Third-party models.** YuNet + SFace are Apache-2.0 from the OpenCV zoo
  and are downloaded, not vendored.

---

## Known traps (details in `HANDOFF.md`)

Every one of these cost real time and looked like a different bug:

1. PAM ordering — a line appended *after* `auth include system-auth` never runs.
2. PAM path-based service names are silently ignored on Fedora → falls to
   `/etc/pam.d/other` → `pam_deny` → looks like a broken module.
3. The enrolling user's CLI and the root-run PAM helper resolving **different
   stores** → silent fallback to the password prompt.
4. `sudo` presents `PAM_USER=root` with no useful candidate owner.
5. A `mean` over accessory probabilities collapsing when the head turns →
   enrolment mislabelled → tag routing broken while still appearing to work.
6. Enrolment starvation: a motionless user got "only 1 usable frame".
7. `pam_exec.so quiet` making failures completely invisible.
8. Enrolling under a `--user` that is not the login name → gallery keyed
   under a name PAM will never ask for.
9. **A pseudo-terminal treated as a remote session** → face auth skipped
   itself on every `sudo` from a terminal window. Safety switches that skip
   work hide their own failures; make the skip verbose.

---

## Definition of done for v1.0

- [ ] `sudo` approves by face, reliably, with no password, with the face in frame
- [ ] Password fallback proven intact on a fresh install
- [ ] `default`, `hat`, `headphones` galleries all present and routing correctly
- [ ] Threshold calibrated from measured genuine scores
- [ ] KDE Plasma verified in a VM (SDDM + lock screen)
- [ ] Howdy disabled or removed
- [ ] 34+ tests green, README accurate, no biometric data in git
- [ ] An honest "what this does not protect against" section in the README
