"""Minimal RRDBNet in plain torch -- enough to load ESRGAN-family checkpoints.

Written by hand instead of pulling BasicSR because BasicSR drags in a compiled
extension and a pinned old torchvision, neither of which installs cleanly on
Windows/py3.12. The architecture is small and stable; the two checkpoints we
care about both use it:

    4x-UltraSharp.pth        num_feat=64, num_block=23, num_grow_ch=32, scale=4
    RealESRGAN_x4plus.pth    same shape, different training data

Both store weights under "params_ema" (4x-UltraSharp may also ship "params");
the loader accepts either and strips a trailing "module." prefix.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch: int = 3, num_out_ch: int = 3, scale: int = 4,
                 num_feat: int = 64, num_block: int = 23, num_grow_ch: int = 32):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


_ESRGAN_FIXED = {
    "model.0": "conv_first",
    "model.1.sub.23": "conv_body",
    "model.3": "conv_up1",
    "model.6": "conv_up2",
    "model.8": "conv_hr",
    "model.10": "conv_last",
}


def _remap_esrgan(sd: dict) -> dict:
    """Translate the original ESRGAN naming into BasicSR naming.

    4x-UltraSharp ships as ``model.0`` / ``model.1.sub.N.RDB1.conv1.0`` -- the
    layout of xinntao's first ESRGAN release, not BasicSR's. Same weights, same
    graph, different key strings, so the graph code stays and only names move.
    """
    import re as _re

    fixed = {
        "model.0": "conv_first",
        "model.1.sub.23": "conv_body",
        "model.3": "conv_up1",
        "model.6": "conv_up2",
        "model.8": "conv_hr",
        "model.10": "conv_last",
    }
    body_re = _re.compile(r"model\.1\.sub\.(\d+)\.(RDB\d)\.(conv\d)\.0\.(\w+)$")
    keep = ("conv_first", "body", "conv_body", "conv_up1", "conv_up2",
            "conv_hr", "conv_last")

    out: dict = {}
    for k, v in sd.items():
        m = body_re.match(k)
        if m:
            i, rdb, conv, tail = m.groups()
            out[f"body.{i}.{rdb.lower()}.{conv}.{tail}"] = v
            continue
        stem, _, param = k.rpartition(".")
        if stem in fixed:
            out[f"{fixed[stem]}.{param}"] = v
            continue
        out[k] = v
    return {k: v for k, v in out.items() if k.split(".")[0] in keep}


def load_model(path: str, device: str = "cuda", fp16: bool = True) -> RRDBNet:
    state = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("params_ema", "params", "state_dict"):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    if not isinstance(state, dict):
        raise RuntimeError(f"{path}: unexpected checkpoint layout")

    state = {k.removeprefix("module."): v for k, v in state.items()}
    if any(k.startswith("model.") for k in state):
        state = _remap_esrgan(state)
    model = RRDBNet()
    missing, unexpected = model.load_state_dict(state, strict=False)
    critical = [k for k in missing if not k.endswith("num_batches_tracked")]
    if critical:
        raise RuntimeError(f"{path}: {len(critical)} missing tensors, first={critical[0]}")
    model.eval()
    model = model.to(device)
    if fp16:
        model = model.half()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.inference_mode()
def upscale_tiled(model: RRDBNet, img: torch.Tensor, tile: int = 256,
                  pad: int = 16) -> torch.Tensor:
    """Run the network over an image in tiles.

    A 512x512 cube face at x4 needs ~1.1 GB of activations in fp16; a 2 MP
    panorama at once does not fit next to anything else on a 12 GB card. Tiles
    with 16 px of context keep the seams invisible because the network's
    receptive field is far smaller than that padding.
    """
    _, _, H, W = img.shape
    out = torch.zeros(1, 3, H * model.scale, W * model.scale,
                      dtype=img.dtype, device=img.device)

    for y in range(0, H, tile):
        for x in range(0, W, tile):
            y1, x1 = min(y + tile, H), min(x + tile, W)
            ys, xs = max(0, y - pad), max(0, x - pad)
            ye, xe = min(H, y1 + pad), min(W, x1 + pad)
            patch = img[:, :, ys:ye, xs:xe]
            res = model(patch).clamp(0, 1)
            s = model.scale
            oy, ox = (y - ys) * s, (x - xs) * s
            oh, ow = (y1 - y) * s, (x1 - x) * s
            out[:, :, y * s:y1 * s, x * s:x1 * s] = res[:, :, oy:oy + oh, ox:ox + ow]
    return out
