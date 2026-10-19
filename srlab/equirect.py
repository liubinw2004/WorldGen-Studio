"""Equirectangular <-> cubemap reprojection in pure torch.

Why cubemap and not 2D tiles: an equirectangular panorama has a 360 degree seam
at x=0 and extreme vertical stretching at the poles. Cutting it into 2D tiles
and upscaling each tile breaks both -- the seam tears, and the top/bottom rows
turn to mush because the model sees a horizontally smeared strip.

A cube face is an ordinary 90 degree perspective image. That is exactly what an
ESRGAN-style network was trained on, so faces upscale cleanly and tile edges
line up afterwards. This is the standard "cut into regions, upscale, stitch"
approach, done in the geometry where stitching actually works.

Angle convention matches WorldGen's ``pano_unit_rays``: phi = atan2(x, z),
theta = asin(y), i.e. direction = (sin(phi)cos(theta), sin(theta), cos(phi)cos(theta)).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# (forward, right, up) per face, in the +X -X +Y -Y +Z -Z order.
FACES = (
    ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
    ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    ((0, 1, 0), (1, 0, 0), (0, 0, 1)),
    ((0, -1, 0), (1, 0, 0), (0, 0, -1)),
    ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),
)


def _as_chw(x: torch.Tensor) -> torch.Tensor:
    """(H,W,3) / (3,H,W) / (1,3,H,W) float in [0,1] -> (1,3,H,W)."""
    if x.dim() == 4:
        return x
    if x.dim() == 3:
        x = x.permute(2, 0, 1) if x.shape[-1] == 3 else x
    return x.unsqueeze(0)


def dirs_to_equirect(d: torch.Tensor) -> torch.Tensor:
    """Directions (..., 3) -> grid_sample coordinates (..., 2) in [-1, 1]."""
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    phi = torch.atan2(x, z)
    theta = torch.asin(y.clamp(-1.0, 1.0))
    u = phi / (2 * math.pi) + 0.5
    v = 0.5 - theta / math.pi
    # Normalised for grid_sample(align_corners=False): x=-1 is the left edge of
    # pixel 0, so continuous pixel p maps to 2*p/W - 1. Feeding u directly as
    # 2u-1 is that expression with p = u*W.
    return torch.stack([u * 2 - 1, v * 2 - 1], dim=-1)


def equirect_to_cubemap(pano: torch.Tensor, face: int,
                        margin: float = 0.06) -> list[torch.Tensor]:
    """Cut an equirect panorama into six (face x face) perspective images.

    ``margin`` samples slightly beyond 90 degrees so the upscaler has real
    context across every shared edge. Without it the model invents independent
    content on the two sides of an edge and the stitch shows as a thin crease.

    The panorama is padded horizontally by wrapping before sampling: an
    equirect map is periodic in longitude, and ``padding_mode="border"`` would
    instead clamp, smearing a visible band down the 360 seam.
    """
    pano = _as_chw(pano).float()
    k = max(2, pano.shape[-1] // 64)
    padded = torch.cat([pano[..., -k:], pano, pano[..., :k]], dim=-1)
    pad_w = padded.shape[-1]

    reach = 1.0 + margin
    lin = torch.linspace(-reach, reach, face, dtype=pano.dtype, device=pano.device)
    vv, uu = torch.meshgrid(lin, lin, indexing="ij")

    out = []
    for fwd, right, up in FACES:
        f = torch.tensor(fwd, dtype=pano.dtype, device=pano.device)
        r = torch.tensor(right, dtype=pano.dtype, device=pano.device)
        u = torch.tensor(up, dtype=pano.dtype, device=pano.device)
        d = f + uu[..., None] * r + vv[..., None] * u
        d = d / d.norm(dim=-1, keepdim=True)
        grid = dirs_to_equirect(d)
        # Rescale longitude from the original width onto the padded width.
        u01 = (grid[..., 0] + 1) / 2
        gx = 2.0 * (u01 * pano.shape[-1] + k) / pad_w - 1.0
        grid = torch.stack([gx, grid[..., 1]], dim=-1).unsqueeze(0)
        out.append(F.grid_sample(padded, grid, mode="bilinear",
                                 padding_mode="border", align_corners=False)[0])
    return out


def _dirs(v: torch.Tensor, u: torch.Tensor, dtype, dev) -> torch.Tensor:
    """Unit directions for normalised panorama fractions v (row) and u (col)."""
    theta = (0.5 - v) * math.pi
    phi = (u - 0.5) * 2 * math.pi
    tt, pp = torch.meshgrid(theta, phi, indexing="ij")
    d = torch.stack([torch.sin(pp) * torch.cos(tt),
                     torch.sin(tt).expand_as(pp),
                     torch.cos(pp) * torch.cos(tt)], dim=-1)
    return d / d.norm(dim=-1, keepdim=True)


def _project(d: torch.Tensor, faces, axes, margin: float) -> torch.Tensor:
    """Blend the two nearest cube faces for every direction in ``d``.

    ``uu``/``vv`` are tangents in the nominal 90 degree face frame, but each
    stored face image covers ``1 + margin`` of that frame (the overhang the
    upscaler needs for context). Dividing by the reach before sampling is what
    keeps the faces registered to the same scale -- without it every face is
    stretched by the margin, the overhangs overlap neighbouring content, and a
    bright line appears along every face boundary.
    """
    dtype, dev = d.dtype, d.device
    reach = 1.0 + margin
    acc = None
    wsum = None
    scores, samples, ramps = [], [], []

    for img, (f, r, u) in zip(faces, axes):
        depth = (d * f).sum(-1).clamp_min(1e-6)
        uu = (d * r).sum(-1) / depth
        vv = (d * u).sum(-1) / depth
        grid = torch.stack([uu / reach, vv / reach], dim=-1).unsqueeze(0)
        samples.append(F.grid_sample(img, grid, mode="bilinear",
                                     padding_mode="border", align_corners=False)[0])
        ramp = (1.0 - torch.maximum(uu.abs(), vv.abs()).clamp(0, 1)) / max(margin, 1e-3)
        ramps.append(ramp.clamp(0, 1))
        scores.append(depth)

    s = torch.stack(scores, dim=0)
    top2 = s.topk(2, dim=0).indices
    keep = torch.zeros_like(s)
    keep.scatter_(0, top2[:1], 1.0)
    keep.scatter_(0, top2[1:2], 1.0)

    for i in range(len(faces)):
        w_i = ramps[i] * keep[i]
        term = samples[i] * w_i.unsqueeze(0)
        acc = term if acc is None else acc + term
        wsum = w_i.unsqueeze(0) if wsum is None else wsum + w_i.unsqueeze(0)

    return acc / wsum.clamp_min(1e-6)


def cubemap_to_equirect(faces: list[torch.Tensor], width: int, height: int,
                        margin: float = 0.06, row_chunk: int = 64,
                        pole_frac: float = 0.03, supersample: int = 4) -> torch.Tensor:
    """Reverse projection with a two-face feathered blend.

    Polar rows need supersampling: one row near the pole spans all 360 degrees,
    so point-sampling that row from a cube corner is wildly anisotropic and
    smears the top and bottom of the panorama. Those bands are rendered at
    ``supersample`` times the resolution and box-filtered back down, which is
    the same thing a mipmapped renderer would do.
    """
    faces = [f.unsqueeze(0) if f.dim() == 3 else f for f in faces]
    dtype, dev = faces[0].dtype, faces[0].device
    axes = [(torch.tensor(f, dtype=dtype, device=dev),
             torch.tensor(r, dtype=dtype, device=dev),
             torch.tensor(u, dtype=dtype, device=dev)) for f, r, u in FACES]

    out = torch.zeros(3, height, width, dtype=dtype, device=dev)
    band = max(1, int(height * pole_frac))

    for y0 in range(0, height, row_chunk):
        y1 = min(y0 + row_chunk, height)
        polar = supersample > 1 and (y1 <= band or y0 >= height - band)
        ss = supersample if polar else 1
        v = (torch.arange(y0 * ss, y1 * ss, device=dev, dtype=dtype) + 0.5) / (height * ss)
        u = (torch.arange(width * ss, device=dev, dtype=dtype) + 0.5) / (width * ss)
        block = _project(_dirs(v, u, dtype, dev), faces, axes, margin)
        if ss > 1:
            block = F.avg_pool2d(block.unsqueeze(0), ss).squeeze(0)
        out[:, y0:y1, :] = block

    return out.clamp(0, 1)

def blend_polar(sr: torch.Tensor, ref: torch.Tensor, frac: float = 0.035) -> torch.Tensor:
    """Cross-fade the extreme top/bottom rows back to a plain resample.

    A cube corner has almost no angular resolution, so the rows nearest the
    poles cannot be recovered by the cube path no matter how much supersampling
    is used -- the information is not there in the representation. Those rows
    are sky or ground in a generated panorama, so falling back to a bicubic
    resample costs nothing and removes the smeared cap entirely.
    """
    if sr.shape != ref.shape:
        raise ValueError(f"shape mismatch {tuple(sr.shape)} vs {tuple(ref.shape)}")
    h = sr.shape[-2]
    band = max(1, int(round(h * frac)))
    ramp = torch.linspace(0, 1, band, dtype=sr.dtype, device=sr.device)
    ramp = 0.5 - 0.5 * torch.cos(ramp * math.pi)      # smoothstep-ish, no hard edge
    w = torch.zeros(h, dtype=sr.dtype, device=sr.device)
    w[:band] = 1.0 - ramp
    w[-band:] = ramp
    w = w[:, None]
    return sr * (1 - w) + ref * w
