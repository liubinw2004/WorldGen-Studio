"""Headless end-to-end run: panorama -> SR (optional) -> depth -> Gaussians.

Unlike demo.py this never starts viser, so it exits and releases the GPU, which
makes it usable as a benchmark harness and as CI for the shim stack.

    T_OUT=outputs/sr_test WG_SR_MODEL=ultrasharp \
        venv/Scripts/python.exe scripts/test_sr_pipeline.py

Env:
    T_MODE   t2s | i2s           (default t2s)
    T_RES    panorama width       (default 1024)
    T_PROMPT prompt for t2s
    T_IMAGE  input image for i2s
    T_OUT    output directory (required)
    WG_SR_MODEL / WG_SR_SCALE / WG_SR_MAX_WIDTH as in run_i2s_patched.py
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

# <root>/scripts/test_sr_pipeline.py -> <root>
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "vendor" / "worldgen_i2s_shim"),
          str(ROOT / "vendor" / "worldgen_sr_shim")):
    sys.path.insert(0, p)

# Every cache/temp path must resolve under the project root before torch loads.
import config as cfg  # noqa: E402

os.environ.update({k: str(v) for k, v in cfg.ENV_VARS.items()})
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402


def main() -> int:
    mode = os.environ.get("T_MODE", "t2s").lower()
    res = int(os.environ.get("T_RES", "1024"))
    prompt = os.environ.get(
        "T_PROMPT",
        "a sunlit alpine meadow with a small wooden cabin, wildflowers, "
        "distant snow peaks, clear blue sky")
    image_path = os.environ.get("T_IMAGE", "").strip()
    out = Path(os.environ["T_OUT"])
    out.mkdir(parents=True, exist_ok=True)

    import worldgen_i2s_shim as shim
    if not shim.apply_patch(64):
        print("[test] FATAL: i2s shim failed", flush=True)
        return 3

    sr_key = os.environ.get("WG_SR_MODEL", "").strip().lower()
    if sr_key and sr_key not in ("0", "off", "none"):
        import worldgen_sr_shim as srshim
        if not srshim.apply_from_env():
            print("[test] WARN: SR shim did not apply", flush=True)
    else:
        print("[test] SR disabled", flush=True)

    from worldgen import WorldGen

    # --- VRAM sampling (nvidia-smi style, counts non-PyTorch allocations too) --
    peak = {"used": 0}
    stop = threading.Event()

    def sampler() -> None:
        while not stop.is_set():
            try:
                free, total = torch.cuda.mem_get_info()
                used = (total - free) / 1024 ** 2
                peak["used"] = max(peak["used"], used)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.25)

    threading.Thread(target=sampler, daemon=True).start()

    # --- record what the panorama producer actually returned -----------------
    from worldgen.worldgen import WorldGen as _WG
    seen: dict = {}
    _pano_fn = _WG.generate_pano

    def _traced(self, *a, **kw):
        t = time.time()
        img = _pano_fn(self, *a, **kw)
        seen["pano"] = [img.width, img.height]
        seen["pano_seconds"] = round(time.time() - t, 1)
        print(f"[test] panorama ready {img.width}x{img.height} "
              f"({seen['pano_seconds']}s)", flush=True)
        return img

    _WG.generate_pano = _traced

    t0 = time.time()
    wg = WorldGen(mode=mode, resolution=res, low_vram=True)
    t_load = time.time() - t0
    print(f"[test] models loaded in {t_load:.1f}s", flush=True)

    image = None
    if mode == "i2s":
        from PIL import Image
        image = Image.open(image_path).convert("RGB")

    t1 = time.time()
    splat = wg.generate_world(prompt=prompt, image=image)
    t_gen = time.time() - t1
    stop.set()

    ply = out / "splat.ply"
    splat.save(str(ply))
    n = int(splat.centers.shape[0])

    stats = {
        "mode": mode,
        "resolution": res,
        "sr_model": sr_key or None,
        "pano": seen.get("pano"),
        "gaussians": n,
        "load_seconds": round(t_load, 1),
        "pano_seconds": seen.get("pano_seconds"),
        "generate_seconds": round(t_gen, 1),
        "total_seconds": round(time.time() - t0, 1),
        "peak_vram_mib": round(peak["used"]),
        "ply_bytes": ply.stat().st_size if ply.exists() else 0,
    }
    (out / "bench.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("[test] DONE " + json.dumps(stats), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
