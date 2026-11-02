"""Standalone model fetch: `python3 -m faceid.models_dl [dest]`."""
from __future__ import annotations

import sys
from pathlib import Path

from .config import resolve_model_dir
from .models import ensure_models, verify_models


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    dest = Path(argv[0]) if argv else resolve_model_dir()
    print(f"faceid models -> {dest}")
    try:
        got = ensure_models(dest)
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    ok = True
    for name, r in got.items():
        print(f"  {name}: {'ok' if r.get('ok') else 'FAILED'}"
              f"{' (downloaded)' if r.get('downloaded') else ''}")
        ok = ok and bool(r.get("ok"))
    for name, r in verify_models(dest).items():
        if not r.get("ok"):
            print(f"  verify: {name} {r}", file=sys.stderr)
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
