# faceid

AI face authentication for Fedora (GNOME **and** KDE Plasma) that keeps
working when you put a hat, a cap, a beanie, a hood or headphones on — and
that can approve `sudo`.

```
faceid enroll --tag default --tag hat --tag headphones   # teach it all your states
faceid enroll --tag auto                                 # or let it decide from what you're wearing
faceid train                                             # learn YOUR hat and YOUR headphones
faceid verify                                            # is it me?
sudo faceid pam install                                  # let it approve sudo
```

---

## Why the usual setup fails with a hat on

The usual Linux setup — **Howdy 3.x** — uses dlib's HOG face detector and
dlib's 2017 ResNet embedding. HOG needs a full, frontal,
fully lit face: a cap brim costs it the forehead, headphones change the head
silhouette, a turned head loses the eyes — and it just stops finding you. Its
single stored template is then compared to nothing.

`faceid` replaces the whole chain:

| stage | Howdy (dlib) | faceid |
|---|---|---|
| detect | HOG — needs a full frontal face | **YuNet CNN** (ONNX, OpenCV DNN), tolerant of occlusion, pose, hats |
| align | 5-point, fragile | OpenCV `alignCrop` similarity transform |
| embed | dlib ResNet v1 (2017) | **SFace CNN** (ONNX) 128-d, trained with heavy augmentation |
| lighting | raw pixels | CLAHE + gamma variants averaged → a dim room matches a bright enrolment |
| templates | **one** per user | **one gallery per tag** — bare, hatted, headphoned |
| accessory info | none | per-user **learned classifier** + geometric priors |
| decision | single comparison | multi-frame voting + consistency (agreement bonus) + margin + liveness |
| IR camera | unused | first-class (see below) |

## The tag system — the actual answer to "works with a hat on"

A **tag** is what you were wearing when you enrolled:

```
default              bare face
hat                  cap / beanie / hood / helmet / beret
headphones           headphones / earbuds / headset / AirPods
hat,headphones       both together
glasses              registered as a tag (gallery routing only)
```

`faceid enroll --tag hat` while wearing the cap, `--tag default` clean, and
then at verify time **every tag is scored and the best one wins**. You are
never forced through one compromise template — whichever you currently look
like is the gallery you get compared against.

`--tag auto` skips the question entirely: it captures, asks the accessory
classifier what you were wearing, names the tag from that answer
(`hat` / `headphones` / `hat,headphones` / `default`) and stores it. Useful
when you just want to add "the way I am right now" without thinking about it.

One verified example of the routing working, from `faceid verify --explain`
against a real enrolment:

```
alice  tag=default   score=0.9843  votes=6  agree=0.00  final=0.9643
alice  tag=hat       score=0.9832  votes=6  agree=1.00  final=1.0332   <- accepted
```

The bare-face gallery actually scored *marginally* higher on similarity, and
would have won on similarity alone. The accessory classifier recognised the
cap, the `hat` tag agreed with it, and the agreement bonus selected the right
gallery. That is the whole design in one output.

### The accessory classifier

Enable it by tagging your enrolments; `faceid train` fits a small
logistic-regression classifier on 64 head-region features (HSV histograms,
Sobel magnitude + orientation histograms, skin fractions on the forehead /
cheeks / ears, ear-side vertical edge energy, crown dark fraction, colour
contrast between covering and face, and more) extracted from *your* frames.

Two independent binary classifiers — `headwear` and `earwear` — so
"hat **and** headphones" is represented correctly rather than as a third
class. It trains in milliseconds from your own data (no 100 MB model, no
shared dataset) and is therefore specific to *your* cap in *your* lighting.

Trained or not, a **geometric prior** always runs: the face is unchanged, the
region around it is not. A cap hides forehead skin and puts a strong
horizontal edge across the crown region; headphone cups add vertical
structure over the ears and remove ear skin. Before any training, the
heuristic already reports `headwear 0.67` on the real capped-face test frame
in this repo's test suite.

The prediction is then used at match time: agreeing with the matched tag adds
`match.accessory_bonus`, contradicting it subtracts `match.accessory_penalty`.
So a hat makes it *more* confident, not less.

## Robustness, in order of how much it actually matters

1. **Enrol the states you actually appear in.** A `hat` gallery beats any
   amount of cleverness applied to a bare-face template.
2. **YuNet + SFace instead of HOG + ResNet-v1** — the biggest structural win,
   especially for occlusion and profile.
3. **Multi-frame voting** (`core.min_votes`, default 3 of ~20 frames).
   One lucky frame can't authenticate anyone.
4. **Illumination-normalised embeddings** — CLAHE and gamma variants averaged,
   so evening lamp-light still matches a daylight enrolment.
5. **Accessory-conditioned routing** — the bonus/penalty above.
6. **Adaptive threshold** — `faceid calibrate --apply` measures *your*
   genuine score and sets the cut-off relative to it.
7. **IR camera support** — Windows-Hello-class laptops expose a monochrome IR
   sensor as an extra `/dev/videoN` (look for a 600×600 `GREY` format). With
   its emitter enabled (`faceid ir`), IR sees
   through a hat brim's shadow, works in a dark room, and a monitor or a
   photo emits no IR at all, which kills replay attacks outright.
   Set `video.prefer_ir = true` once the emitter is configured.
8. **Basic liveness** — a frozen frame or a photo taped over the lens shows
   zero micro-motion and is rejected. **Honest scope: this is a speed bump,
   not a certified anti-spoof defence.** Requiring IR is the real fix.

### What "always works" honestly means

No face authenticator is 100% — a beard, 40 kg of weight change, a broken
camera or a pitch-black room will defeat any of them. What this design
guarantees is that **you are never locked out**: faceid is installed into PAM
as `auth sufficient`, never `required`, so a failure only ever falls through
to the normal password prompt. Plus:

* `core.abort_if_ssh = true` — never tries the camera over SSH;
* a hard 12 s `timeout` in the PAM helper — a wedged camera cannot block sudo;
* `core.disabled = true` in the config is a one-line kill switch that needs
  no PAM edit at all;
* `sudo faceid pam uninstall` restores every PAM file byte-for-byte.

## Install

```bash
cd ~/faceid
sudo ./install.sh                 # sudo + sudo-i, wrappers, models, SELinux
sudo ./install.sh --all           # also GDM / SDDM / polkit / screensaver / su
sudo ./install.sh --replace-howdy # and disable the old pam_howdy lines
sudo ./install.sh --dry-run       # show exactly what would change, write nothing
```

Then **keep a root shell open** and test in a second terminal:

```bash
sudo true          # should recognise you, or fall back to the password
faceid pam status
sudo faceid pam uninstall      # if anything is wrong
```

The installer writes only `auth sufficient` lines, backs up every file it
touches to `<file>.faceid-bak`, refuses to write a stack that would lose its
password fallback, re-reads and verifies after writing, and rolls back on
mismatch.

### What KDE Plasma needs that GNOME doesn't

`--all` covers both, but they are different PAM stacks:

| what you want to unlock | GNOME | KDE Plasma |
|---|---|---|
| `sudo` in a terminal | `sudo` | `sudo` |
| graphical admin prompt (polkit) | `polkit-1` | `polkit-1` |
| login screen | `gdm-password` | `sddm` (+ `sddm-greeter`) |
| lock screen | `gdm-password` | `kscreenlocker` |
| `su` | `su`, `su-l` | `su`, `su-l` |

Two Plasma-specific caveats, stated plainly:

* The `sddm`, `sddm-greeter` and `kscreenlocker` service names are written
  from the Plasma PAM layout but **have not been exercised on a real Plasma
  session** by the author. `sudo`, `polkit-1`, `su`, `gdm-password` and
  `system-auth` have. Run `./install.sh --all --dry-run` first and read the
  previews, which show the exact file you are about to get.
* Plasma's lock screen runs its PAM stack as the session user, not as root,
  so that user must be able to open the camera. Desktop logins normally get
  an ACL on `/dev/video*` automatically, but if your lock screen reports no
  camera, add yourself to the `video` group (`sudo usermod -aG video $USER`)
  and log out and back in.

## Commands

```
faceid doctor                full environment report (cameras, models, SELinux, PAM env)
faceid devices               probe every /dev/videoN and say what it really is
faceid models --download     fetch/verify the two ONNX models (~39 MB)
faceid enroll --tag hat      capture and store a tagged gallery
faceid enroll --tag auto     let the accessory classifier name the tag
faceid train                 fit the accessory classifier from tagged enrolments
faceid verify [--explain]    authenticate; exit 0 on success
faceid test [--explain]      identify without applying the threshold
faceid calibrate --apply     measure your genuine score and set the threshold from it
faceid list / remove         inspect and prune the store
faceid ir                    IR sensors, and how to turn the emitter on
faceid config --set video.prefer_ir=true
faceid selinux [--build-policy]
faceid pam install|uninstall|status|log
faceid auth [--why]          the entry point pam_exec.so calls
```

### When sudo still asks for a password

`faceid pam log` is the answer, and it exists because this failure mode is
silent by construction: `pam_exec.so quiet` writes nothing to syslog, so a
failing `sufficient` module is indistinguishable from a module that never ran.

```
$ faceid pam log
2026-11-02T14:38:35+08:00 rc=1 user=alice ruser=alice service=sudo
    REJECT no usable frame with a face was captured (0 frames read)
```

That one line tells you the helper ran, as whom, and why it said no. If the
log is *absent* after a `sudo` attempt, then `pam_exec` never executed the
helper at all, and the problem is the PAM wiring rather than the face.

Config lives at `/etc/faceid/config.toml` (system) overlaid by
`~/.config/faceid/config.toml`. Every key, with its default and an
explanation, is in `faceid/config.py`.

## Files it installs

```
/usr/local/lib/faceid/            the package
/usr/local/bin/faceid             CLI wrapper
/usr/local/bin/faceid-pam         PAM helper (hard 12 s timeout)
/usr/share/faceid/models/*.onnx   YuNet + SFace
/var/lib/faceid/gallery/          your enrolments, 0700, owned by you
/etc/faceid/config.toml           defaults
```

## Models and licence

Both models come from the official **OpenCV model zoo** and are Apache-2.0:
`face_detection_yunet_2023mar.onnx` (232 KB) and
`face_recognition_sface_2021dec.onnx` (37 MB). They run on OpenCV's own DNN
engine, so there is **no extra runtime dependency** beyond `python3-opencv`
and `python3-numpy` — both already present from Fedora's repos. Checksums are
verified on download.

## Development

```bash
/usr/bin/python3 tests/test_faceid.py     # 34 tests, pytest not required
python3 -m pytest tests/ -v               # if you do have pytest
```

The camera-dependent tests skip themselves when no ground-truth frame is
available, so the suite is safe on a headless box. To run them, drop any
photo of a face at `tests/fixtures/face.png` (gitignored — a face image is
biometric data and must never be committed) or set `FACEID_GT_FRAME`:

```bash
FACEID_GT_FRAME=/path/to/a/face.png /usr/bin/python3 tests/test_faceid.py
```

Two notes for Fedora specifically:

* use **`/usr/bin/python3`**, not a venv — a virtualenv frequently has no
  `cv2`, and the system `python3-opencv` already provides everything needed;
* `tests/pam_harness.py` drives a **real PAM stack through libpam via ctypes**
  — no compiler, no `pamtester`, no root needed for the read-only mode:

  ```bash
  python3 tests/pam_harness.py --service sudo --user $USER
  sudo python3 tests/pam_harness.py --service sudo --probe   # + marker line
  ```

  Its conversation function refuses to answer any prompt, which is exactly
  what `sudo -n` does, so a success means face auth alone satisfied the stack.
  Use `--probe` to prove whether `pam_exec.so` is executing commands at all;
  do not trust a path-based `--stack` on Fedora, where `pam_start` silently
  falls through to `/etc/pam.d/other` and produces a failure that looks like
  a broken module but is really a missing service file.
