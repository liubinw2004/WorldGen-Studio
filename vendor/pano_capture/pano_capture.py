"""Capture the intermediate panorama as a PNG so the super-resolution stage can
be evaluated offline.

WorldGen keeps the panorama in memory only -- ``generate_pano`` returns a
PIL.Image that is immediately consumed by ``_generate_world`` and never written.
For SR comparison work we need that exact image, so this module wraps the method
and dumps it when ``WG_PANO_OUT`` is set. Unset -> completely inert.

No WorldGen source file is modified.
"""
from __future__ import annotations

import os
import time


def install() -> bool:
    out = os.environ.get("WG_PANO_OUT", "").strip()
    if not out:
        return False

    try:
        import worldgen.worldgen as wg
    except Exception as exc:  # noqa: BLE001
        print(f"[pano-capture] import failed: {exc!r}", flush=True)
        return False

    orig = wg.WorldGen.generate_pano
    if getattr(orig, "_pano_capture", False):
        return True

    def _gp(self, *args, **kwargs):
        t0 = time.time()
        img = orig(self, *args, **kwargs)
        try:
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            img.save(out)
            print(f"[pano-capture] saved {img.size[0]}x{img.size[1]} -> {out} "
                  f"({time.time() - t0:.1f}s)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[pano-capture] save failed: {exc!r}", flush=True)
        return img

    _gp._pano_capture = True
    wg.WorldGen.generate_pano = _gp
    print(f"[pano-capture] armed, target={out}", flush=True)
    return True
