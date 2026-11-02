#!/bin/sh
# faceid - AI face authentication with hat/headphone awareness.
#
#   ./install.sh                 install for sudo (+ wrappers, models, SELinux)
#   ./install.sh --all           also wire up GDM/SDDM/polkit/screensaver
#   ./install.sh --replace-howdy disable the existing pam_howdy lines
#   ./install.sh --dry-run       show exactly what would change
#
# Everything it edits is backed up to <file>.faceid-bak and restored by
#   sudo faceid pam uninstall
set -eu

SRC="$(cd "$(dirname "$0")" && pwd)"
ARGS=""

for a in "$@"; do
    case "$a" in
        --all) ARGS="$ARGS --services all" ;;
        *)     ARGS="$ARGS $a" ;;
    esac
done

if [ "$(id -u)" -ne 0 ]; then
    echo "install.sh must run as root:  sudo $0 $*" >&2
    exit 1
fi

echo "faceid installer"
echo "  source   : $SRC"
echo "  user     : ${SUDO_USER:-root}"
echo

# The models are ~39 MB and are downloaded once, as the calling user, so the
# checksums land in a directory we can then install from.
if ! ls "$SRC"/models/*.onnx >/dev/null 2>&1; then
    echo "== downloading AI models (~39 MB) =="
    PYTHONPATH="$SRC" python3 -m faceid.models_dl "$SRC/models"
fi

PYTHONPATH="$SRC" exec python3 -m faceid.install $ARGS
