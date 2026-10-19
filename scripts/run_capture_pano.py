"""Run demo.py with the VRAM shim AND panorama capture enabled.

Used to produce the reference panorama that the super-resolution stage is
evaluated against. Identical to run_i2s_patched.py plus one extra runtime hook.

Usage:
    WG_PANO_OUT=D:\\mm\\outputs\\pano_ref\\pano_1024.png \
    python D:\\mm\\scripts\\run_capture_pano.py -p "<prompt>" -o <out> -r 1024 --low_vram
"""
from __future__ import annotations

import os
import runpy
import sys

# <root>/scripts/run_capture_pano.py -> <root>
ROOT = Path(__file__).resolve().parent.parent
SHIM_DIR = str(ROOT / "vendor" / "worldgen_i2s_shim")
CAPTURE_DIR = str(ROOT / "vendor" / "pano_capture")
DEMO_PY = str(ROOT / "WorldGen" / "demo.py")


def main() -> int:
    for d in (SHIM_DIR, CAPTURE_DIR):
        sys.path.insert(0, d)

    try:
        import worldgen_i2s_shim as shim
    except Exception as exc:  # noqa: BLE001
        print(f"[run_capture_pano] FATAL: cannot import VRAM shim: {exc!r}", flush=True)
        return 3
    if not shim.apply_patch(int(os.environ.get("I2S_SHIM_BATCH", "64"))):
        print("[run_capture_pano] FATAL: VRAM shim did not apply", flush=True)
        return 3

    # Must be imported AFTER the shim so the pipeline classes are already wrapped.
    try:
        import pano_capture
        if not pano_capture.install():
            print("[run_capture_pano] FATAL: WG_PANO_OUT not set or hook failed", flush=True)
            return 4
    except Exception as exc:  # noqa: BLE001
        print(f"[run_capture_pano] FATAL: capture hook failed: {exc!r}", flush=True)
        return 4

    sys.argv = [DEMO_PY] + sys.argv[1:]
    runpy.run_path(DEMO_PY, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
