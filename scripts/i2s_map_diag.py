"""Diagnostic: confirm the WorldGen i2s panorama-mapping tensor sizes and prove
the ``worldgen_i2s_shim`` makes ``map_image_to_pano`` run within 12 GB VRAM.

This only loads the DA-2 depth model (not the diffusion pipeline), so it is a
fast, isolated reproduction of the exact OOM site:
    worldgen/utils/general_utils.py::batch_nearest_dot  (torch.mm -> (8192,N_src))

Run:
    D:\\mm\\venv\\Scripts\\python.exe D:\\mm\\scripts\\i2s_map_diag.py [<image>]
"""
from __future__ import annotations

import glob
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg  # noqa: E402

os.environ.update(cfg.ENV_VARS)
sys.path.insert(0, str(ROOT / "vendor" / "worldgen_i2s_shim"))


def find_demo() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    hits = glob.glob(str(cfg.HF_HUB_CACHE / "models--mit-han-lab--svdq-int4-flux.1-fill-dev"
                         / "snapshots" / "*" / "demo.jpg"))
    return hits[0]


def main() -> int:
    import torch
    from PIL import Image
    from worldgen.utils.general_utils import resize_img, resize_img_and_rays, map_image_to_pano
    from worldgen.pano_depth import build_depth_model, pred_depth
    import worldgen_i2s_shim as shim

    demo = find_demo()
    print(f"image = {demo}", flush=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = resize_img(Image.open(demo).convert("RGB"))
    print(f"resized input = {img.size}", flush=True)

    batch = int(os.environ.get("I2S_SHIM_BATCH", "64"))
    shim.apply_patch(batch)

    model = build_depth_model(dev)
    t0 = time.time()
    preds = pred_depth(model, img)
    torch.cuda.synchronize() if dev.type == "cuda" else None
    print(f"pred_depth: rgb={tuple(preds['rgb'].shape)} rays={tuple(preds['rays'].shape)} "
          f"distance={tuple(preds['distance'].shape)}  ({time.time() - t0:.1f}s)", flush=True)

    rgb_r, rays_r = resize_img_and_rays(preds["rgb"].float(), preds["rays"], 1024, 2048)
    n_src = rgb_r.shape[0] * rgb_r.shape[1]
    print(f"resize_img_and_rays -> {tuple(rgb_r.shape)}  => N_src={n_src:,} "
          f"(upstream dense sim = {8192 * n_src * 4 / 1024 ** 3:.1f} GiB)", flush=True)

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    pano_img, mask_img = map_image_to_pano(preds, device=dev)
    torch.cuda.synchronize() if dev.type == "cuda" else None
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2 if dev.type == "cuda" else 0.0
    print(f"map_image_to_pano: OK pano={pano_img.size} mask={mask_img.size} "
          f"time={dt:.2f}s peak_alloc={peak:.1f} MiB", flush=True)
    print("DIAG: PASS (mapping completes with shim)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
