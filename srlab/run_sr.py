"""Upscale an equirect panorama through cubemap faces.

    python -m srlab.run_sr --pano pano_1024.png --model ultrasharp \
        --out pano_2048_ultrasharp.png --target-width 2048

Thin CLI over ``srlab.upscale`` so the standalone tool and the in-process
WorldGen patch cannot drift apart.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from srlab.upscale import MODELS, upscale_pano  # noqa: E402

# Accept the old full filenames as aliases so earlier commands keep working.
ALIASES = {
    "4x-ultrasharp": "ultrasharp",
    "realesrgan_x4plus": "realesrgan",
    "real-esrgan": "realesrgan",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pano", required=True)
    ap.add_argument("--model", default="ultrasharp")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-width", type=int, default=2048)
    ap.add_argument("--margin", type=float, default=0.06)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()

    key = ALIASES.get(args.model.lower(), args.model.lower())
    if key not in MODELS:
        ap.error(f"unknown model {args.model!r}, pick from {sorted(MODELS)}")

    from PIL import Image

    pano = Image.open(args.pano)
    print(f"[sr] source {pano.width}x{pano.height} -> target width {args.target_width}, "
          f"model={MODELS[key]['name']}", flush=True)
    img, stats = upscale_pano(
        pano, key, target_width=args.target_width,
        margin=args.margin, tile=args.tile,
        device="cpu" if args.fp32 else None,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)
    stats["out"] = args.out
    Path(args.out).with_suffix(".json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[sr] done in {stats['seconds']}s, peak {stats['peak_mib']:.0f} MiB "
          f"-> {args.out} ({stats['target'][0]}x{stats['target'][1]})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
