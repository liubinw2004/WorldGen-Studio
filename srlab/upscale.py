"""Reusable panorama super-resolution: equirect -> cubemap -> x4 net -> equirect.

This is the single implementation shared by the standalone CLI
(``python -m srlab.run_sr``) and the in-process WorldGen patch
(``vendor/worldgen_sr_shim``). Keeping one copy matters: the seam geometry is
fiddly enough that two drifting implementations would silently produce
different-looking output from the CLI and the app.

Pipeline
--------
1. cut the equirect into 6 perspective faces, each with a 6% margin so the
   network sees real context across face borders and the feathered reprojection
   has spare pixels to blend;
2. run the 4x RRDB network per face, tiled;
3. area-downsample each face from 4x to the size the target panorama needs
   (x4-then-shrink is sharper than asking a 2x net for 2x);
4. reproject to equirect with two-face feathering, then fall back to bicubic
   in the top/bottom 3.5% where a cube face simply runs out of angular
   resolution.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from srlab.equirect import (blend_polar, cubemap_to_equirect,
                            equirect_to_cubemap)
from srlab.rrdb import load_model, upscale_tiled

# <root>/srlab/upscale.py -> <root>
_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("WORLDGEN_SR_DIR") or (_ROOT / "models" / "sr"))

MODELS = {
    "ultrasharp": {
        "name": "4x-UltraSharp",
        "path": str(MODEL_DIR / "4x-UltraSharp.pth"),
        "label": "4x-UltraSharp",
        "note": "锐利、边缘干净，点云观感更好",
    },
    "realesrgan": {
        "name": "RealESRGAN_x4plus",
        "path": str(MODEL_DIR / "RealESRGAN_x4plus.pth"),
        "label": "Real-ESRGAN x4plus",
        "note": "柔和、噪点少，接近照片质感",
    },
}

# Default ceiling on the upscaled panorama width. 4096 x 2048 = 8.4M pixels,
# which would put ~8.4M Gaussians (~470 MB ply) on a 12 GB card -- possible, but
# the depth geometry behind it is interpolated anyway (see the note in
# ``plan_width``). 2048 (t2s 1024 -> 2048, 2.1M Gaussians) is the sweet spot.
MAX_WIDTH = 2048


def to_tensor(img: Image.Image) -> torch.Tensor:
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def to_pil(t: torch.Tensor) -> Image.Image:
    a = (t.squeeze(0).clamp(0, 1).permute(1, 2, 0).float().cpu().numpy() * 255.0 + 0.5)
    return Image.fromarray(a.astype(np.uint8))


def plan_width(src_w: int, scale: int = 2, target_width: int | None = None,
               max_width: int = MAX_WIDTH) -> int:
    """Decide the output width: requested, else src*scale, clamped to max_width.

    Note on what a higher panorama resolution does *not* buy: DA-2 (the depth
    model) internally resizes its input to a fixed ~600k-pixel budget and
    bilinearly stretches the result back, so the geometry behind the extra
    pixels is interpolated, not measured. What does improve is the RGB the
    Gaussians are painted with -- ``convert_rgbd_to_gs`` emits one point per
    panorama pixel, so 2x width means 4x points carrying genuinely sharper
    texture.
    """
    want = int(target_width) if target_width else src_w * int(scale)
    want = max(src_w, min(want, int(max_width)))
    # Equirect panoramas are 2:1; keep the width even so h = w/2 stays integral.
    return want - (want % 2)


def upscale_pano(pano: Image.Image, model_key: str = "ultrasharp",
                 target_width: int | None = None, scale: int = 2,
                 margin: float = 0.06, tile: int = 256,
                 device: str | None = None,
                 max_width: int = MAX_WIDTH) -> tuple[Image.Image, dict]:
    """Upscale an equirect panorama. Returns ``(image, stats)``.

    The model is loaded, used and released inside this call so that callers
    never inherit a resident 610 MiB network.
    """
    key = (model_key or "ultrasharp").lower()
    if key not in MODELS:
        raise ValueError(f"unknown SR model {model_key!r}, expected {sorted(MODELS)}")

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    src = pano.convert("RGB")
    src_w, src_h = src.size
    tgt_w = plan_width(src_w, scale, target_width, max_width)
    tgt_h = tgt_w // 2

    if tgt_w <= src_w:
        return src, {"skipped": True, "source": [src_w, src_h],
                     "target": [src_w, src_h], "model": key}

    # 90 degrees of panorama == one quarter of its width. Both the source face
    # and the destination face are derived from that, so the reprojection is
    # resolution-independent.
    face_src = max(64, src_w // 4)
    face_dst = tgt_w // 4

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    x = to_tensor(src).to(dev)
    faces = equirect_to_cubemap(x, face_src, margin=margin)

    model = load_model(MODELS[key]["path"], device=dev, fp16=(dev == "cuda"))
    up = []
    with torch.inference_mode():
        for f in faces:
            r = upscale_tiled(model, f.unsqueeze(0).to(next(model.parameters()).dtype),
                              tile=tile)
            r = F.interpolate(r, size=(face_dst, face_dst), mode="area")
            up.append(r.float()[0])

    out = cubemap_to_equirect(up, tgt_w, tgt_h, margin=margin)
    ref = F.interpolate(x, size=(tgt_h, tgt_w), mode="bicubic",
                        align_corners=False).clamp(0, 1)
    out = blend_polar(out, ref[0])

    img = to_pil(out)

    # Release everything the upscaler touched. The caller (WorldGen) is about
    # to hand the panorama to the depth model and then the Gaussian builder, and
    # on a 12 GB card those need the room.
    del model, faces, up, x, out, ref
    if dev == "cuda":
        torch.cuda.empty_cache()

    peak = torch.cuda.max_memory_allocated() / 1024 ** 2 if dev == "cuda" else 0.0
    stats = {
        "skipped": False,
        "model": key,
        "model_name": MODELS[key]["name"],
        "source": [src_w, src_h],
        "target": [tgt_w, tgt_h],
        "scale": round(tgt_w / src_w, 2),
        "seconds": round(time.time() - t0, 1),
        "peak_mib": round(peak, 1),
        "tile": tile,
    }
    return img, stats
