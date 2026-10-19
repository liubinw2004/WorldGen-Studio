"""Panorama super-resolution patch for WorldGen (no source files modified).

Where it hooks
--------------
``WorldGen.generate_world`` is just ``generate_pano`` followed by
``_generate_world``. Patching ``generate_pano`` -- rather than
``generate_world`` -- is deliberate: the existing VRAM shim already wraps
``generate_world`` to move the depth model off the GPU, and two wrappers around
the same method would either fight over ordering or force one of them to
re-implement the pipeline. Patching the panorama producer keeps the two shims
orthogonal and composable.

VRAM choreography
-----------------
On a 12 GB card the diffusion stack (nunchaku INT4 transformer ~6.6 GB, plus
whatever the VAE pulls in) must be gone before the upscaler loads, even though
the upscaler only needs ~610 MiB. The order is intentionally serial and
mutually exclusive:

    diffusion (6.6 GB)  ->  released  ->  SR (0.6 GB)  ->  released
                        ->  depth DA-2 (~1.3 GB)  ->  Gaussian build

Nothing overlaps, so the peak never grows; the upscaled panorama simply costs
an extra second and feeds the depth model at 2x.

Usage::

    import worldgen_sr_shim as sr
    sr.apply_patch("ultrasharp", scale=2, max_width=2048)
    # then run demo.py as usual
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

# <root>/vendor/worldgen_sr_shim/worldgen_sr_shim.py -> <root>
ROOT = Path(__file__).resolve().parent.parent.parent
for p in (str(ROOT), str(ROOT / "vendor" / "worldgen_sr_shim")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _release_diffusion(worldgen_obj) -> None:
    """Move the diffusion pipeline to CPU and hand the VRAM back to the driver.

    ``pano_gen_model`` is a diffusers pipeline built with
    ``enable_model_cpu_offload()``. ``.to("cpu")`` on such a pipeline normally
    detaches the offload hooks first; older diffusers versions raise instead, so
    the fallback walks ``components`` manually. Either way the point is the same
    -- after this returns, the transformer is not on the GPU.
    """
    pipe = getattr(worldgen_obj, "pano_gen_model", None)
    if pipe is None:
        return
    freed = False
    try:
        pipe.to("cpu")
        freed = True
    except Exception as exc:  # noqa: BLE001
        print(f"[sr-shim] pipe.to(cpu) failed ({exc!r}), moving components", flush=True)
        comps = getattr(pipe, "components", None)
        if isinstance(comps, dict):
            for name, comp in comps.items():
                if hasattr(comp, "to"):
                    try:
                        comp.to("cpu")
                        freed = True
                    except Exception as inner:  # noqa: BLE001
                        print(f"[sr-shim]   {name}.to(cpu) failed: {inner!r}", flush=True)
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:  # noqa: BLE001
        pass
    if torch.cuda.is_available():
        free = torch.cuda.mem_get_info()[0] / 1024 ** 2
        print(f"[sr-shim] diffusion released={freed}, GPU free={free:.0f} MiB", flush=True)


def apply_patch(model_key: str = "ultrasharp", scale: int = 2,
                max_width: int = 2048, tile: int = 256,
                margin: float = 0.06) -> bool:
    """Patch ``WorldGen.generate_pano`` to upscale its output.

    ``model_key`` is ``ultrasharp`` or ``realesrgan``. Returns False if WorldGen
    could not be imported (the caller should then run without SR rather than
    crash: a missing shim must never be worse than no shim).
    """
    # Imported lazily: this module may be loaded before torch/CUDA is warm.
    try:
        from worldgen.worldgen import WorldGen
    except Exception as exc:  # noqa: BLE001
        print(f"[sr-shim] could not import WorldGen: {exc!r}", flush=True)
        return False

    from srlab.upscale import MODELS, upscale_pano

    key = (model_key or "").lower()
    if key not in MODELS:
        print(f"[sr-shim] unknown model {model_key!r}, expected {sorted(MODELS)}", flush=True)
        return False

    orig = WorldGen.generate_pano
    if getattr(orig, "_sr_shim", False):
        return True

    print(f"[sr-shim] pano super-resolution -> {MODELS[key]['name']}, "
          f"scale={scale}x, max width={max_width}", flush=True)

    def _generate_pano(self, prompt: str = "", image=None):
        pano = orig(self, prompt, image)
        try:
            _release_diffusion(self)
            out, stats = upscale_pano(
                pano, key, scale=scale, max_width=max_width,
                tile=tile, margin=margin,
            )
            if stats.get("skipped"):
                print(f"[sr-shim] panorama already {pano.width}px wide, "
                      f"SR skipped", flush=True)
            else:
                print(f"[sr-shim] pano {stats['source'][0]}x{stats['source'][1]} -> "
                      f"{stats['target'][0]}x{stats['target'][1]} in "
                      f"{stats['seconds']}s (peak {stats['peak_mib']:.0f} MiB, "
                      f"{MODELS[key]['name']})", flush=True)
            return out
        except Exception as exc:  # noqa: BLE001
            # A failed upscale must degrade to the plain panorama, not sink the
            # whole run: the diffusion step already cost 40 s.
            import traceback
            traceback.print_exc()
            print(f"[sr-shim] upscale failed ({exc!r}); falling back to the "
                  f"{pano.width}px panorama", flush=True)
            return pano

    _generate_pano._sr_shim = True
    WorldGen.generate_pano = _generate_pano
    return True


def apply_from_env() -> bool:
    """Convenience entry point driven by WG_SR_MODEL / WG_SR_SCALE / WG_SR_MAX_WIDTH."""
    key = os.environ.get("WG_SR_MODEL", "").strip().lower()
    if not key or key in ("0", "off", "none"):
        return False
    try:
        scale = int(os.environ.get("WG_SR_SCALE", "2"))
    except ValueError:
        scale = 2
    try:
        max_w = int(os.environ.get("WG_SR_MAX_WIDTH", "2048"))
    except ValueError:
        max_w = 2048
    return apply_patch(key, scale=scale, max_width=max_w)
