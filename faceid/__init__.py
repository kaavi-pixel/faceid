"""faceid - robust, AI face authentication for Fedora (GNOME + KDE Plasma).

Pipeline
--------
capture  ->  detect (YuNet CNN)  ->  align  ->  embed (SFace CNN, 128-d)
         ->  accessory analysis (learned per-user classifier)
         ->  tag-aware gallery match + multi-frame voting
         ->  PAM (sudo / GDM / SDDM / polkit / screensaver)

Design goal: keep working when the user wears a hat, headphones, glasses,
a hood, or is in a dark room.  See README.md for the robustness notes.
"""

__version__ = "1.0.0"
APP = "faceid"
