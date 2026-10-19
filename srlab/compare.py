"""Build a side-by-side comparison sheet plus seam metrics for pano upscalers.

    python -m srlab.compare --src pano_1024.png \
        --sr ultrasharp=out_ultra.png realesrgan=out_real.png \
        --out compare_sheet.png

The sheet has one row of full-panorama thumbnails and three rows of 1:1 crops
chosen where upscalers actually differ: the 360 seam, the polar cap, and a
detail-dense region. Bilinear (what the pipeline does today) is row/column one.

Seam metric: for a 360 panorama, column 0 and column W-1 are neighbours, so a
correct result has |col0 - colW-1| close to the average adjacent-column
difference. Anything much larger means the stitch is visible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def adjacent_diff(a: np.ndarray) -> float:
    """Mean absolute difference between horizontally neighbouring columns."""
    return float(np.abs(np.diff(a, axis=1)).mean())


def wrap_diff(a: np.ndarray) -> float:
    """Difference across the 360 degree wrap (last column vs first)."""
    return float(np.abs(a[:, 0] - a[:, -1]).mean())


def seam_rows(a: np.ndarray, quarters: int = 4) -> dict:
    """Check vertical creases at the face boundaries as well as the wrap."""
    h, w = a.shape[:2]
    out = {}
    for q in range(1, quarters):
        x = w * q // quarters
        out[f"x={x}"] = float(np.abs(a[:, x - 1] - a[:, x]).mean())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--sr", nargs="+", required=True, help="name=path ...")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", type=int, default=320)
    args = ap.parse_args()

    src = Image.open(args.src).convert("RGB")
    variants = [("bilinear (current)", None)]
    for item in args.sr:
        name, _, path = item.partition("=")
        variants.append((name, path))

    imgs, metrics = [], {}
    for name, path in variants:
        if path is None:
            ref = Image.open(args.src)
            imgs.append((name, src.resize((ref.width * 2, ref.height * 2), Image.BILINEAR)))
        else:
            imgs.append((name, Image.open(path).convert("RGB")))

    tgt_w, tgt_h = imgs[0][1].size
    for name, im in imgs:
        a = np.asarray(im.resize((tgt_w, tgt_h), Image.BILINEAR), dtype=np.float32)
        metrics[name] = {
            "size": list(im.size),
            "adjacent_col_diff": round(adjacent_diff(a), 3),
            "wrap_diff": round(wrap_diff(a), 3),
            "face_seams": {k: round(v, 2) for k, v in seam_rows(a).items()},
        }

    cols = len(imgs)
    cw = args.crop
    pad, label_h, thumb_h = 10, 22, 180
    sheet_w = cols * cw + (cols + 1) * pad
    sheet_h = pad + label_h + thumb_h + 3 * (label_h + cw + pad)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (16, 18, 24))
    draw = ImageDraw.Draw(sheet)

    spots = [
        ("360 seam", 0, tgt_h // 2 - cw // 2),
        ("polar cap", tgt_w // 2 - cw // 2, 0),
        ("detail", tgt_w // 3, tgt_h // 3),
    ]

    y = pad
    for i, (name, im) in enumerate(imgs):
        x = pad + i * (cw + pad)
        thumb = im.resize((int(cw), int(cw * im.height / im.width)), Image.LANCZOS)
        sheet.paste(thumb, (x, y + label_h))
        draw.text((x + 2, y + 4), name, fill=(220, 226, 236))

    y = pad + label_h + thumb_h + pad
    for spot_name, sx, sy in spots:
        draw.text((pad, y), f"{spot_name}", fill=(150, 160, 178))
        y += label_h
        for i, (name, im) in enumerate(imgs):
            x = pad + i * (cw + pad)
            im2 = im if im.size == (tgt_w, tgt_h) else im.resize((tgt_w, tgt_h), Image.LANCZOS)
            box = (sx, sy, sx + cw, sy + cw)
            sheet.paste(im2.crop(box), (x, y))
        y += cw + pad

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    Path(args.out).with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[cmp] sheet -> {args.out}")
    for name, m in metrics.items():
        seams = m["face_seams"]
        worst = max(seams.items(), key=lambda kv: kv[1]) if seams else ("-", 0)
        print(f"[cmp] {name:22s} wrap={m['wrap_diff']:6.2f} "
              f"adj={m['adjacent_col_diff']:6.2f} worst_face_seam={worst[0]}:{worst[1]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
