"""Run WorldGen's demo.py with the runtime shims applied.

Two independent shims, both runtime-only (no WorldGen source file is touched):

* ``vendor/worldgen_i2s_shim``   -- rebinds ``batch_nearest_dot`` to a
  memory-bounded chunked version and frees VRAM around T5 encoding;
* ``vendor/worldgen_sr_shim``    -- upscales the generated panorama before the
  depth/Gaussian stage, when ``WG_SR_MODEL`` is set.

They compose because they patch different methods: the first wraps
``WorldGen.generate_world``, the second wraps ``WorldGen.generate_pano``.

Usage (same args as demo.py):
    python D:\\mm\\scripts\\run_i2s_patched.py -i <image> -p "<prompt>" \
        -o <out> -r 1024 --low_vram --save_scene

Env:
    I2S_SHIM_BATCH  query chunk size for batch_nearest_dot (default 64)
    WG_SR_MODEL     ultrasharp | realesrgan | off (default off)
    WG_SR_SCALE     panorama upscale factor (default 2)
    WG_SR_MAX_WIDTH width ceiling (default 2048)
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

# <root>/scripts/run_i2s_patched.py -> <root>
ROOT = Path(__file__).resolve().parent.parent
SHIM_DIR = ROOT / "vendor" / "worldgen_i2s_shim"
SR_SHIM_DIR = ROOT / "vendor" / "worldgen_sr_shim"
DEMO_PY = ROOT / "WorldGen" / "demo.py"


def main() -> int:
    if not DEMO_PY.exists():
        print(f"[run_i2s_patched] FATAL: WorldGen not found at {DEMO_PY}. "
              f"Run install.bat first.", flush=True)
        return 4

    sys.path.insert(0, str(SHIM_DIR))
    try:
        import worldgen_i2s_shim as shim
    except Exception as exc:  # noqa: BLE001
        print(f"[run_i2s_patched] FATAL: cannot import shim: {exc!r}", flush=True)
        return 3

    batch = int(os.environ.get("I2S_SHIM_BATCH", "64"))
    if not shim.apply_patch(batch):
        print("[run_i2s_patched] FATAL: shim failed to patch batch_nearest_dot", flush=True)
        return 3

    # Super-resolution is optional. A failure here must not abort the run --
    # without it we simply get the plain panorama we would have had anyway.
    sr_key = os.environ.get("WG_SR_MODEL", "").strip().lower()
    if sr_key and sr_key not in ("0", "off", "none"):
        sys.path.insert(0, SR_SHIM_DIR)
        try:
            import worldgen_sr_shim as srshim
            if not srshim.apply_from_env():
                print("[run_i2s_patched] WARN: SR shim did not apply; "
                      "continuing at base resolution", flush=True)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"[run_i2s_patched] WARN: SR shim crashed ({exc!r}); "
                  f"continuing at base resolution", flush=True)

    # Forward every CLI arg to the unmodified demo.py.
    sys.argv = [DEMO_PY] + sys.argv[1:]
    runpy.run_path(DEMO_PY, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
