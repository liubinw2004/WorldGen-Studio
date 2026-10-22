"""Panoramic (equirectangular) video renderer for WorldGen Studio.

This module turns an *existing* Gaussian-splat scene (``outputs/studio/<id>/splat.ply``)
into a 2:1 equirectangular MP4 that a browser can play back inside a sphere.

It is **not** a generative video model. Every frame is a real geometry render of
the point cloud that the text->pano->depth->splat pipeline already produced, so
the result has true parallax, zero VRAM cost and sub-second-per-frame latency.
That choice is deliberate: this machine has 12 GB of VRAM, and a text-to-video
model (Wan / CogVideoX) both needs quantisation to fit and emits *perspective*
video, not a 360 panorama.

Design notes
------------
The naive per-frame cost (recompute spherical projection + argsort + 5 disc
writes at 2048x1024) measures ~476 ms, which makes a 6 s clip take over a
minute. Instead each scene is projected **once** into a cached
``SplatScene`` whose arrays are already depth-sorted far->near; a pure rotation
of the camera then degenerates to an integer column shift
``u = (u0 + du) % W`` (~71 ms/frame). Only ``dolly`` / ``crane`` move the
camera, and those re-project and re-sort per frame (~300-500 ms/frame).

Coordinate system
-----------------
The PLY is OpenCV-style (**+y down**) and the equirectangular convention below
is the one the browser viewer already uses; both are copied verbatim from
``_probe/splat_render.py`` / ``app/static/world.js`` because a sign flip here
silently mirrors the whole world::

    r     = |p|
    theta = arccos(-p.y / r)      # 0 = zenith (top row of the image)
    phi   = atan2(p.x, p.z)       # -pi..pi, +z = image centre
    u     = ((phi / 2pi) + 0.5) * W
    v     = (theta / pi) * H

Public API
----------
``load_ply(path) -> (centers, rgb)``
``SplatScene``                      cached, resolution-specific projection
``PATHS``                           camera-move presets (the ``path`` param)
``generate_video(scene_dir, opts, progress_cb, cancel_flag) -> dict``
``encode_mp4(frames, out_path, fps, crf=20)``

PyAV is imported lazily so that a missing ``av`` package never prevents the
Flask server (or anything else importing this module) from starting.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
SH_C0: float = 0.28209479177387814
"""3DGS SH DC -> linear RGB scale (``rgb = SH_C0 * f_dc + 0.5``)."""

MAX_POINTS: int = 3_000_000
"""Hard ceiling. Beyond this the per-frame numpy gathers blow up in RAM."""

DOLLY_MAX_M: float = 0.8
"""Hard ceiling for dolly/crane camera translation.

Measured: at 1.0 m the splat coverage falls from 100% to 43% because the cloud
is a single shell captured from one viewpoint -- the disoccluded area behind
near geometry has no points to show. 0.8 m keeps the frame solid.
"""

DOLLY_TRAVEL_M: float = 0.4
"""Translation the dolly/crane presets use by default.

The scene is a *regular equirect lattice* (one Gaussian per pixel of the source
1024x512 panorama), so any re-projection resamples that lattice and moire rings
appear once the angular shift approaches the sample spacing. Measured
high-frequency energy on the pond surface, relative to the still frame:
0.2 m -> 1.8x, 0.4 m -> 2.4x, 0.5 m -> 3.1x, 0.6 m -> 4.1x, 0.8 m -> 7.2x.
0.4 m is the largest travel that still reads as a clean push-in (and it is a
quarter of the 1.52 m median scene depth, so the parallax is visible). Callers
may override it up to :data:`DOLLY_MAX_M` with ``opts["travel"]``.
"""

ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO_DIR: Path = ROOT / "outputs" / "videos"

VIDEO_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,158}\.mp4$")
"""Whitelist for ``GET /api/world/video-file/<name>`` (no separators, no ..)."""

_TWO_PI = 2.0 * math.pi

# Day / night grade, mirrored from the fragment shader in app/static/world.js so
# a rendered clip looks like what the live viewer shows.
_TINT_DAY = np.array([1.05, 1.03, 0.97], np.float32)
_TINT_NIGHT = np.array([0.52, 0.66, 1.10], np.float32)
_TINT_GOLD = np.array([1.22, 1.02, 0.76], np.float32)


class PanoVideoError(RuntimeError):
    """Raised for every user-visible panorama-video failure (message is Chinese)."""


class PanoCancelled(PanoVideoError):
    """Raised inside the frame generator when the caller sets the cancel flag."""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _smoothstep(edge0: float, edge1: float, x):
    """GLSL-style smoothstep. Works for scalars and numpy arrays."""
    t = np.clip((np.asarray(x, np.float32) - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _even(n: int) -> int:
    """Round an int down to the nearest even number (yuv420p requirement)."""
    n = int(n)
    return n - (n % 2)


def _shift_rows(a: np.ndarray, dy: int) -> np.ndarray:
    """Shift ``a`` along axis 0 by ``dy``, clamping at the edges (no wrap)."""
    if dy == 0:
        return a
    out = np.empty_like(a)
    if dy > 0:
        out[:dy] = a[:1]
        out[dy:] = a[:-dy]
    else:
        k = -dy
        out[:-k] = a[k:]
        out[-k:] = a[-1:]
    return out


def _shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift a HxW or HxWxC array: rows clamp, columns wrap (equirect seam)."""
    out = a
    if dy:
        out = _shift_rows(out, dy)
    if dx:
        out = np.roll(out, dx, axis=1)
    return out


# 8-neighbourhood, 4-axis first so a hole prefers a direct neighbour.
_NEIGH8: Tuple[Tuple[int, int], ...] = (
    (0, 1), (0, -1), (1, 0), (-1, 0),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
)


def disc_offsets(radius: float) -> List[Tuple[int, int]]:
    """Integer pixel offsets inside a disc of ``radius``.

    radius 1.0 -> 5 samples, 1.5 -> 9, 2.0 -> 13. Centre is returned **last**
    because later writes win, so the core pixel is never overwritten by a rim
    sample of the same point.
    """
    r = int(math.ceil(max(0.0, float(radius))))
    out: List[Tuple[int, int]] = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy * dy + dx * dx <= radius * radius:
                out.append((dy, dx))
    out.sort(key=lambda t: -abs(t[0]) - abs(t[1]))
    return out


# --------------------------------------------------------------------------- #
# PLY loading
# --------------------------------------------------------------------------- #
_PLY_DTYPES: Dict[str, str] = {
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
    "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
}


def _read_ply_header(path: Path) -> Tuple[List[Tuple[str, str]], int, int]:
    """Parse a binary-little-endian PLY header.

    Returns ``(props, vertex_count, data_offset)`` where ``props`` is an ordered
    list of ``(type, name)`` for the vertex element.
    """
    buf = bytearray()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(4096)
            if not chunk:
                raise PanoVideoError(f"PLY 头部不完整：{path.name}")
            buf.extend(chunk)
            if b"end_header" in buf:
                break
            if len(buf) > (1 << 20):
                raise PanoVideoError(f"PLY 头部异常，超过 1MB 仍未结束：{path.name}")

    end = buf.index(b"end_header")
    nl = buf.find(b"\n", end)
    if nl < 0:
        raise PanoVideoError(f"PLY 头部缺少换行：{path.name}")
    header = bytes(buf[:nl]).decode("ascii", "replace")

    fmt = ""
    element = ""
    count = 0
    props: List[Tuple[str, str]] = []
    for raw in header.splitlines():
        line = raw.strip()
        if not line:
            continue
        tok = line.split()
        if tok[0] == "format":
            fmt = tok[1] if len(tok) > 1 else ""
        elif tok[0] == "element":
            element = tok[1] if len(tok) > 1 else ""
            if element == "vertex" and len(tok) > 2:
                try:
                    count = int(tok[2])
                except ValueError:
                    raise PanoVideoError(f"PLY 顶点数不是整数：{line}") from None
        elif tok[0] == "property" and element == "vertex":
            if len(tok) >= 3 and tok[1] != "list":
                props.append((tok[1], tok[2]))
            elif len(tok) >= 2 and tok[1] == "list":
                raise PanoVideoError(
                    "PLY 顶点的 list 属性暂不支持（本工具只读取定长标量属性）")

    if fmt not in ("binary_little_endian",):
        raise PanoVideoError(f"只支持 binary_little_endian PLY，实际为 {fmt or '?'}")
    if count <= 0 or not props:
        raise PanoVideoError(f"PLY 里没有顶点数据：{path.name}")
    return props, count, nl + 1


def load_ply(ply_path) -> Tuple[np.ndarray, np.ndarray]:
    """Read a Gaussian-splat PLY into ``(centers float32 Nx3, rgb uint8 Nx3)``.

    The files written by this project carry **no** ``red/green/blue``; colour
    lives in the 3DGS SH DC coefficients ``f_dc_0/1/2``. They are decoded with
    the standard ``SH_C0 * f_dc + 0.5`` and clipped. Plain RGB properties are
    accepted as a fallback so the function also works on converted clouds.

    Raises:
        PanoVideoError: on a missing/unreadable file, an unsupported format, or
            a cloud larger than :data:`MAX_POINTS`.
    """
    path = Path(ply_path)
    if not path.exists():
        raise PanoVideoError(f"点云文件不存在：{path}")

    props, count, offset = _read_ply_header(path)
    if count > MAX_POINTS:
        raise PanoVideoError(
            f"点云共 {count:,} 个点，超过 {MAX_POINTS:,} 的上限。"
            "Python 端渲染这么多点会耗尽内存，请先降低场景点数。")

    try:
        dtype = np.dtype([(name, _PLY_DTYPES[t]) for t, name in props])
    except KeyError as exc:  # pragma: no cover - exotic PLY
        raise PanoVideoError(f"不支持的 PLY 属性类型：{exc}") from None

    with open(path, "rb") as fh:
        fh.seek(offset)
        raw = np.fromfile(fh, dtype=dtype, count=count)

    if raw.size < count:
        raise PanoVideoError(f"PLY 数据长度不足：期望 {count:,} 个点，读到 {raw.size:,}")

    names = raw.dtype.names or ()
    for need in ("x", "y", "z"):
        if need not in names:
            raise PanoVideoError(f"PLY 缺少坐标属性 {need}：{path.name}")

    centers = np.empty((count, 3), np.float32)
    centers[:, 0] = raw["x"]
    centers[:, 1] = raw["y"]
    centers[:, 2] = raw["z"]

    rgb = np.empty((count, 3), np.uint8)
    if all(f"f_dc_{i}" in names for i in range(3)):
        for i in range(3):
            lin = np.asarray(raw[f"f_dc_{i}"], np.float32) * SH_C0 + 0.5
            np.clip(lin, 0.0, 1.0, out=lin)
            rgb[:, i] = (lin * 255.0 + 0.5).astype(np.uint8)
    elif all(c in names for c in ("red", "green", "blue")):
        rgb[:, 0] = raw["red"]
        rgb[:, 1] = raw["green"]
        rgb[:, 2] = raw["blue"]
    else:
        raise PanoVideoError(
            f"PLY 既没有 f_dc_0/1/2 也没有 red/green/blue，无法取色：{path.name}")

    return centers, rgb


# --------------------------------------------------------------------------- #
# Camera + camera-move presets
# --------------------------------------------------------------------------- #
@dataclass
class Camera:
    """A frame's camera state.

    ``yaw`` / ``pitch`` are radians; ``offset`` is a translation of the camera
    in the PLY's own frame (metres). ``colors`` optionally overrides the tinted
    colour table that :meth:`SplatScene.render` would otherwise compute.
    """

    yaw: float = 0.0
    pitch: float = 0.0
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    colors: Optional[np.ndarray] = None


PATHS: Dict[str, Dict[str, object]] = {
    "still": {
        "label": "静止", "desc": "机位不动，只有昼夜与风",
        "reproject": False, "slow": False,
    },
    "orbit": {
        "label": "环绕", "desc": "偏航转一圈，首尾帧相接可无缝循环",
        "reproject": False, "slow": False,
    },
    "handheld": {
        "label": "手持", "desc": "轻微低频抖动，加一点呼吸感",
        "reproject": False, "slow": False,
    },
    "dolly": {
        "label": "推近",
        "desc": (f"沿 +z 推进 0~{DOLLY_TRAVEL_M} m 再回落，产生视差"
                 f"（上限 {DOLLY_MAX_M} m；较慢）"),
        "reproject": True, "slow": True,
    },
    "crane": {
        "label": "升降",
        "desc": (f"垂直起降 0~{DOLLY_TRAVEL_M} m 再回落"
                 f"（上限 {DOLLY_MAX_M} m；较慢）"),
        "reproject": True, "slow": True,
    },
}


def sample_camera(path_name: str, t01: float, seconds: float,
                  travel: Optional[float] = None) -> Camera:
    """Camera state at normalised time ``t01`` in ``[0, 1)``.

    ``still`` / ``orbit`` / ``handheld`` are pure rotations, which the renderer
    turns into integer pixel shifts. ``dolly`` / ``crane`` translate the camera
    and therefore force a per-frame re-projection.

    The translating paths ease out and back (``0.5 - 0.5*cos``) instead of
    running one way, which keeps the clip seamless under ``video.loop = true``
    while never exceeding ``travel`` (clamped to :data:`DOLLY_MAX_M`).
    """
    name = str(path_name or "still").lower()
    if name not in PATHS:
        raise PanoVideoError(
            f"未知运镜 {name!r}，可用：{', '.join(sorted(PATHS))}")
    t01 = float(min(1.0, max(0.0, t01)))
    seconds = max(0.5, float(seconds))

    if name == "still":
        return Camera()
    if name == "orbit":
        return Camera(yaw=_TWO_PI * t01)
    if name == "handheld":
        # Frequencies are integer multiples of 1/seconds so the last frame
        # lands back on the first one.
        t = t01 * seconds
        w1 = _TWO_PI * 2.0 / seconds
        w2 = _TWO_PI * 5.0 / seconds
        yaw = 0.026 * math.sin(w1 * t) + 0.013 * math.sin(w2 * t + 1.1)
        pitch = 0.012 * math.sin(w1 * t + 2.0) + 0.006 * math.sin(w2 * t + 0.4)
        return Camera(yaw=yaw, pitch=pitch)

    k = 0.5 - 0.5 * math.cos(_TWO_PI * t01)
    dist = DOLLY_TRAVEL_M if travel is None else float(travel)
    dist = min(DOLLY_MAX_M, max(0.0, dist))
    if name == "dolly":
        return Camera(offset=(0.0, 0.0, dist * k))
    return Camera(offset=(0.0, -dist * k, 0.0))  # crane: -y is up


# --------------------------------------------------------------------------- #
# SplatScene - per-scene precomputation
# --------------------------------------------------------------------------- #
class SplatScene:
    """A point cloud projected once for a fixed output resolution.

    Construction does everything that does not depend on the frame: colour
    decode, depth sort (far -> near), spherical projection to integer pixel
    coordinates, luminance, wind phase. Everything after that is O(N) gather
    work, so a rotation costs ~71 ms instead of ~476 ms at 2048x1024.
    """

    def __init__(self, scene_id: str, ply_path, width: int, height: int,
                 radius: float = 1.0) -> None:
        self.scene_id = str(scene_id)
        self.ply_path = Path(ply_path)
        self.width = _even(int(width))
        self.height = _even(int(height))
        self.radius = float(max(1.0, radius))
        self.offsets = disc_offsets(self.radius)

        centers, rgb = load_ply(self.ply_path)
        r = np.sqrt(np.einsum("ij,ij->i", centers, centers))
        keep = r > 1e-3
        if not keep.all():
            centers = centers[keep]
            rgb = rgb[keep]
            r = r[keep]
        if centers.shape[0] == 0:
            raise PanoVideoError(f"场景 {self.scene_id} 的点云全在原点，无法渲染")

        # far -> near: later writes win, so nearer splats overwrite farther ones
        order = np.argsort(-r, kind="stable")
        self.centers = np.ascontiguousarray(centers[order], np.float32)
        self.depth = np.ascontiguousarray(r[order], np.float32)
        self.rgb_float = np.ascontiguousarray(
            rgb[order].astype(np.float32) / 255.0, np.float32)
        self.count = int(self.centers.shape[0])
        del centers, rgb, r, order

        x = self.centers[:, 0]
        y = self.centers[:, 1]
        z = self.centers[:, 2]
        d = self.depth
        theta = np.arccos(np.clip(-y / d, -1.0, 1.0))
        phi = np.arctan2(x, z)
        # Normalised projection, kept so render() can serve a different
        # resolution without redoing the trigonometry.
        self.u_norm = np.ascontiguousarray(phi / _TWO_PI + 0.5, np.float32)
        self.v_norm = np.ascontiguousarray(theta / np.pi, np.float32)
        self.u0 = self._to_u(self.u_norm, self.width)
        self.v0 = self._to_v(self.v_norm, self.height)
        self.lum = np.ascontiguousarray(
            self.rgb_float @ np.array([0.299, 0.587, 0.114], np.float32),
            np.float32)
        # World "up" is -y (OpenCV frame); wind only moves what is above ground.
        self.lift = np.ascontiguousarray(
            np.clip((-y - 0.15) / 5.0, 0.0, 1.0), np.float32)
        rng = np.random.default_rng(0x57A17)
        self.phase = (rng.random(self.count, np.float32) * _TWO_PI).astype(np.float32)
        self._tint_light: Optional[float] = None
        self._tint: Optional[np.ndarray] = None

    # -- projection helpers ------------------------------------------------ #
    @staticmethod
    def _to_u(u_norm: np.ndarray, width: int) -> np.ndarray:
        return np.minimum((u_norm * width).astype(np.int32), width - 1)

    @staticmethod
    def _to_v(v_norm: np.ndarray, height: int) -> np.ndarray:
        return np.minimum((v_norm * height).astype(np.int32), height - 1)

    # -- colour ------------------------------------------------------------ #
    def tinted(self, light: float) -> np.ndarray:
        """Per-point RGB for a day/night level, ``(N, 3) uint8``.

        Mirrors ``FRAG`` in app/static/world.js: night is blue, day is nearly
        neutral, and the brightest points (lanterns, lit windows) keep a warm
        gold glow after dark. The last result is memoised because a ``still``
        clip with a constant ``light`` would otherwise recompute it 96 times.
        """
        light = float(min(1.0, max(0.0, light)))
        if self._tint is not None and self._tint_light is not None \
                and abs(self._tint_light - light) < 5e-4:
            return self._tint

        s = float(_smoothstep(0.0, 0.85, light))
        base = _TINT_NIGHT + (_TINT_DAY - _TINT_NIGHT) * s
        warm = (_smoothstep(0.42, 0.95, self.lum) * (1.0 - light) * 0.85)[:, None]
        tint = base[None, :] * (1.0 - warm) + _TINT_GOLD[None, :] * warm
        c = self.rgb_float * tint
        glow = (_smoothstep(0.5, 1.0, self.lum) * (1.0 - light))[:, None]
        c = c * (1.0 - glow) + (c * 1.22 + 0.03) * glow
        c *= 0.92 + 0.10 * light
        np.clip(c, 0.0, 1.0, out=c)
        out = (c * 255.0 + 0.5).astype(np.uint8)
        self._tint_light = light
        self._tint = out
        return out

    # -- wind -------------------------------------------------------------- #
    def wind_offsets(self, t: float, amount: float, width: int) -> np.ndarray:
        """Screen-space jitter, ``(N,) int32`` columns.

        Rather than re-projecting world positions (which would defeat the whole
        cache), each point carries a random phase and is nudged horizontally by
        ``A*sin(omega*t + phase)`` weighted by its height, so only the canopy and
        the lanterns move. The amplitude is expressed in pixels and scales with
        the output width, because 1 px covers half as much sky at 2048 wide.
        """
        amount = float(max(0.0, min(1.0, amount)))
        if amount <= 0.0:
            return np.zeros(self.count, np.int32)
        amp = amount * (1.5 + 2.5 * (float(width) / 1024.0))
        omega = 1.9 * _TWO_PI
        jitter = np.sin(omega * t + self.phase + self.centers[:, 2] * 0.7)
        jitter *= self.lift * amp
        return np.rint(jitter).astype(np.int32)

    def animate(self, t: float, light: float, wind: float,
                width: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Per-frame animation for a static camera: ``(du, colors)``.

        ``du`` is an ``int32 (N,)`` array of column offsets (zeros when
        ``wind`` is 0) and ``colors`` is the tinted ``(N, 3) uint8`` table.
        ``render()`` consumes exactly this pair on its fast path.
        """
        w = int(width or self.width)
        colors = self.tinted(light)
        if not wind:
            return np.zeros(self.count, np.int32), colors
        return self.wind_offsets(t, float(wind), w), colors

    # -- rendering ---------------------------------------------------------- #
    def render(self, cam: Optional[Camera] = None, W: Optional[int] = None,
               H: Optional[int] = None, radius: Optional[float] = None,
               wind: Optional[np.ndarray] = None) -> np.ndarray:
        """Rasterise one equirectangular frame, ``(H, W, 3) uint8``.

        Args:
            cam: camera state; defaults to a still camera at light 0.62.
            W, H: output size; defaults to the resolution this scene was built
                for. A different size falls back to re-deriving u/v from the
                cached normalised projection (no trigonometry).
            radius: splat disc radius in pixels; defaults to this scene's.
            wind: optional ``(N,) int32`` column jitter from :meth:`animate`.

        Raises:
            PanoVideoError: if the colour table or the offsets do not match the
                cloud (never silently returns a black frame).
        """
        cam = cam or Camera()
        W = _even(int(W or self.width))
        H = _even(int(H or self.height))
        rad = float(radius) if radius else self.radius
        colors = cam.colors if cam.colors is not None else self.tinted(0.62)
        if colors.shape[0] != self.count:
            raise PanoVideoError(
                f"颜色表 {colors.shape[0]} 与点数 {self.count} 不一致，无法渲染")

        off = np.asarray(cam.offset, np.float32)
        if not off.any():
            u, v = self._project_rotation(cam, W, H, wind)
        else:
            u, v, colors = self._project_translated(off, cam, W, H, colors)

        offsets = self.offsets if abs(rad - self.radius) < 1e-6 else disc_offsets(rad)
        frame = self._splat(u, v, colors, W, H, offsets)
        if frame is None:  # pragma: no cover - defensive
            raise PanoVideoError("渲染结果为空帧")
        return frame

    def _project_rotation(self, cam: Camera, W: int, H: int,
                          wind: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Pure rotation: integer shift of the cached projection, no re-sort."""
        if W == self.width and H == self.height:
            u0, v0 = self.u0, self.v0
        else:
            u0 = self._to_u(self.u_norm, W)
            v0 = self._to_v(self.v_norm, H)

        du = -int(round(float(cam.yaw) / _TWO_PI * W))
        dv = -int(round(float(cam.pitch) / math.pi * H))
        if wind is not None and wind.any():
            u = (u0 + du + wind) % W
            u = u.astype(np.int32, copy=False)
        elif du:
            u = (u0 + du) % W
        else:
            u = u0
        if dv:
            v = np.clip(v0 + dv, 0, H - 1).astype(np.int32, copy=False)
        else:
            v = v0
        return u, v

    def _project_translated(self, off: np.ndarray, cam: Camera, W: int, H: int,
                            colors: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Camera moved: true re-projection + depth re-sort (dolly / crane)."""
        p = self.centers - off
        r = np.sqrt(np.einsum("ij,ij->i", p, p))
        np.maximum(r, 1e-3, out=r)
        theta = np.arccos(np.clip(-p[:, 1] / r, -1.0, 1.0))
        phi = np.arctan2(p[:, 0], p[:, 2])
        du = float(cam.yaw) / _TWO_PI * W
        dv = float(cam.pitch) / math.pi * H
        u = ((phi / _TWO_PI + 0.5) * W - du).astype(np.int32) % W
        v = np.clip(((theta / math.pi) * H - dv).astype(np.int32), 0, H - 1)

        # Nothing else re-sorts depth, so do it here: far -> near again.
        order = np.argsort(-r, kind="stable")
        return u[order], v[order], colors[order]

    def _splat(self, u: np.ndarray, v: np.ndarray, colors: np.ndarray,
               W: int, H: int, offsets: Sequence[Tuple[int, int]]) -> np.ndarray:
        """Paint disc sprites, then close 1 px seams with one 3x3 pass."""
        flat = np.zeros((H * W, 3), np.uint8)
        hit = np.zeros(H * W, np.uint8)
        v = v.astype(np.int32, copy=False)
        u = u.astype(np.int32, copy=False)
        for dy, dx in offsets:
            vv = v + dy
            if dy:
                vv = np.clip(vv, 0, H - 1)
            uu = u + dx
            if dx:
                uu = uu % W
            idx = vv * W + uu
            flat[idx] = colors
            hit[idx] = 1

        img = flat.reshape(H, W, 3)
        holes = hit.reshape(H, W) == 0
        if holes.any():
            img = _fill_holes(img, holes)
        return img


def _fill_holes(img: np.ndarray, holes: np.ndarray) -> np.ndarray:
    """Copy the nearest filled neighbour into every empty pixel (one pass)."""
    have = ~holes
    acc = np.zeros_like(img)
    got = np.zeros_like(holes)
    for dy, dx in _NEIGH8:
        nb_rgb = _shift(img, dy, dx)
        nb_has = _shift(have, dy, dx)
        take = holes & nb_has & ~got
        if take.any():
            acc[take] = nb_rgb[take]
            got |= take
    out = img.copy()
    out[got] = acc[got]
    return out


# --------------------------------------------------------------------------- #
# Scene cache (max 2 entries - one cloud is 100-250 MB of float32)
# --------------------------------------------------------------------------- #
_CACHE_LOCK = threading.Lock()
_CACHE: "OrderedDict[Tuple[str, int, int, float], SplatScene]" = OrderedDict()
_CACHE_MAX = 2


def get_scene(scene_id: str, ply_path, width: int, height: int,
              radius: Optional[float] = None) -> SplatScene:
    """Return a cached :class:`SplatScene`, building it on first use.

    The cache key is ``(scene_id, width, height, radius)``: the cached projection
    is expressed in pixels *and* the disc offsets are baked into the scene, so a
    different radius needs its own entry. Oldest entries are evicted beyond two.
    """
    width = _even(int(width))
    height = _even(int(height))
    rad = float(radius) if radius else max(1.0, width / 1024.0)
    key = (str(scene_id), width, height, round(rad, 4))
    with _CACHE_LOCK:
        scene = _CACHE.get(key)
        if scene is not None:
            _CACHE.move_to_end(key)
            return scene

    scene = SplatScene(scene_id, ply_path, width, height, rad)

    with _CACHE_LOCK:
        _CACHE[key] = scene
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
    return scene


def clear_cache() -> None:
    """Drop every cached scene (used by tests / after a re-generation)."""
    with _CACHE_LOCK:
        _CACHE.clear()


# --------------------------------------------------------------------------- #
# MP4 encoding (PyAV -> H.264 / yuv420p)
# --------------------------------------------------------------------------- #
def _import_av():
    """Import PyAV lazily so a missing package cannot break ``import server``."""
    try:
        import av  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - ImportError or a broken wheel
        raise PanoVideoError(
            "生成全景视频需要 PyAV（自带 ffmpeg/libx264，无需系统另装 ffmpeg）。"
            "请执行： venv\\Scripts\\python.exe -m pip install \"av>=13.0.0\""
            f"（原始错误：{exc}）") from exc
    import av
    return av


def encode_mp4(frames: Iterable[np.ndarray], out_path, fps: int = 24,
               crf: int = 20, preset: str = "veryfast") -> int:
    """Encode an iterable of ``(H, W, 3) uint8`` frames into a browser MP4.

    Frames are consumed lazily so a 10 s 2048x1024 clip never sits in RAM as a
    whole list. The output is ``avc1`` + ``yuv420p`` + ``+faststart``, which is
    what Chrome/Safari need for ``<video>`` and ``VideoTexture`` playback.

    Args:
        frames: iterable of RGB uint8 arrays, all the same even H x W.
        out_path: destination ``.mp4``.
        fps: frames per second.
        crf: x264 quality (lower = bigger/better); 20 is visually lossless here.
        preset: x264 speed preset.

    Returns:
        The number of frames written.

    Raises:
        PanoVideoError: on an empty stream, odd dimensions, or a bad shape.
    """
    av = _import_av()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fps = int(fps)

    it: Iterator[np.ndarray] = iter(frames)
    try:
        first = next(it)
    except StopIteration:
        raise PanoVideoError("没有任何帧可以编码") from None

    if first.ndim != 3 or first.shape[2] != 3:
        raise PanoVideoError(f"帧格式必须是 HxWx3，实际为 {first.shape}")
    H, W = int(first.shape[0]), int(first.shape[1])
    if H % 2 or W % 2:
        raise PanoVideoError(f"yuv420p 要求宽高为偶数，实际 {W}x{H}")

    if out_path.exists():
        try:
            out_path.unlink()
        except OSError as exc:
            raise PanoVideoError(f"无法覆盖旧视频 {out_path.name}：{exc}") from exc

    container = av.open(str(out_path), "w", options={"movflags": "+faststart"})
    written = 0
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = W
        stream.height = H
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(int(crf)), "preset": preset}

        for img in _chain_first(first, it, H, W):
            frame = av.VideoFrame.from_ndarray(
                np.ascontiguousarray(img, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
            written += 1

        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()

    if written == 0:  # pragma: no cover - first frame already counted
        raise PanoVideoError("没有任何帧被写入视频")
    return written


def _chain_first(first: np.ndarray, rest: Iterator[np.ndarray],
                 H: int, W: int) -> Iterator[np.ndarray]:
    """Yield ``first`` then ``rest``, validating every frame's shape."""
    yield first
    for img in rest:
        if img.shape[0] != H or img.shape[1] != W:
            raise PanoVideoError(
                f"帧尺寸不一致：期望 {W}x{H}，收到 {img.shape[1]}x{img.shape[0]}")
        yield img


# --------------------------------------------------------------------------- #
# Filenames / sidecars
# --------------------------------------------------------------------------- #
def video_filename(scene_id: str, path_name: str,
                   when: Optional[datetime] = None) -> str:
    """``<first 40 chars of scene id>-<path>-<timestamp>.mp4`` (path-safe)."""
    when = when or datetime.now()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", str(scene_id))[:40].strip("-.") or "scene"
    safe_path = re.sub(r"[^a-z0-9]+", "", str(path_name).lower())[:12] or "still"
    return f"{safe}-{safe_path}-{when:%Y%m%d-%H%M%S}.mp4"


def is_valid_video_name(name: str) -> bool:
    """True if ``name`` is a plain ``*.mp4`` file name we are willing to serve."""
    return bool(name) and ".." not in name and bool(VIDEO_NAME_RE.match(name))


def sidecar_path(mp4_path) -> Path:
    """Metadata sidecar that sits next to a rendered clip."""
    return Path(mp4_path).with_suffix(".json")


# --------------------------------------------------------------------------- #
# Top-level job
# --------------------------------------------------------------------------- #
def generate_video(scene_dir, opts: Optional[dict] = None,
                   progress_cb: Optional[Callable[[int, int], None]] = None,
                   cancel_flag: Optional[Callable[[], bool]] = None,
                   out_dir=None) -> dict:
    """Render and encode one panorama clip for a scene directory.

    Args:
        scene_dir: ``outputs/studio/<scene_id>`` (must contain ``splat.ply``).
        opts: ``path`` (see :data:`PATHS`), ``seconds``, ``fps``, ``width``,
            ``height``, ``light_from``, ``light_to``, ``wind``, ``radius``,
            ``crf``.
        progress_cb: called as ``progress_cb(frames_done, frames_total)`` after
            every encoded frame.
        cancel_flag: polled before each frame; when it returns True a
            :class:`PanoCancelled` is raised and the partial file removed.
        out_dir: destination directory; defaults to ``outputs/videos``.

    Returns:
        ``{name, url, bytes, frames, fps, width, height, seconds, scene, path,
        elapsed, created}`` - ``url`` is the server route for the clip.
    """
    opts = dict(opts or {})
    scene_dir = Path(scene_dir)
    scene_id = scene_dir.name or "scene"
    ply = scene_dir / "splat.ply"
    if not ply.exists():
        raise PanoVideoError(f"场景 {scene_id} 里没有 splat.ply，无法生成视频")

    path_name = str(opts.get("path") or "still").lower()
    if path_name not in PATHS:
        raise PanoVideoError(
            f"未知运镜 {path_name!r}，可用：{', '.join(sorted(PATHS))}")

    try:
        fps = int(opts.get("fps") or 24)
        width = _even(int(opts.get("width") or 1024))
        height = _even(int(opts.get("height") or (width // 2)))
        seconds = float(opts.get("seconds") or 6.0)
    except (TypeError, ValueError) as exc:
        raise PanoVideoError(f"参数不是数字：{exc}") from None

    if not 1.0 <= seconds <= 20.0:
        raise PanoVideoError(f"时长必须在 1~20 秒之间，收到 {seconds:g}")
    if not 12 <= fps <= 30:
        raise PanoVideoError(f"帧率必须在 12~30 之间，收到 {fps}")
    if width not in (1024, 2048):
        raise PanoVideoError(f"宽度只支持 1024 或 2048，收到 {width}")
    if height * 2 != width:
        raise PanoVideoError(f"全景视频必须是 2:1，宽度 {width} 时高度应为 {width // 2}")

    light_from = float(opts.get("light_from", 0.62))
    light_to = float(opts.get("light_to", light_from))
    wind_raw = opts.get("wind", 0.6)
    wind = 0.0 if wind_raw is False else (0.6 if wind_raw is True
                                         else float(wind_raw or 0.0))
    wind = float(min(1.0, max(0.0, wind)))
    crf = int(opts.get("crf") or 20)
    radius = opts.get("radius")
    raw_travel = opts.get("travel", None)
    try:
        travel = DOLLY_TRAVEL_M if raw_travel in (None, "") else float(raw_travel)
    except (TypeError, ValueError):
        raise PanoVideoError(f"travel 必须是数字，收到 {raw_travel!r}") from None
    travel = min(DOLLY_MAX_M, max(0.0, travel))

    total = max(1, int(round(seconds * fps)))
    scene = get_scene(scene_id, ply, width, height,
                      float(radius) if radius else None)

    out_dir = Path(out_dir) if out_dir else DEFAULT_VIDEO_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    created = datetime.now()
    name = video_filename(scene_id, path_name, created)
    out_path = out_dir / name
    started = time.perf_counter()

    def frames() -> Iterator[np.ndarray]:
        reproject = bool(PATHS[path_name]["reproject"])
        for i in range(total):
            if cancel_flag is not None and cancel_flag():
                raise PanoCancelled("已取消")
            t01 = i / float(total)
            light = light_from + (light_to - light_from) * t01
            cam = sample_camera(path_name, t01, seconds, travel)
            if reproject:
                # A moved camera changes the depth order anyway, so the wind
                # jitter (which only makes sense for a fixed viewpoint) is
                # skipped -- real parallax already supplies the motion.
                cam.colors = scene.tinted(light)
                wind_px = None
            else:
                wind_px, cam.colors = scene.animate(
                    t01 * seconds, light, wind, width)
            yield scene.render(cam, width, height, scene.radius, wind_px)
            if progress_cb is not None:
                progress_cb(i + 1, total)
            # numpy holds the GIL for the whole frame; yield it so the Flask
            # request threads can still answer the progress poll.
            time.sleep(0)

    try:
        written = encode_mp4(frames(), out_path, fps=fps, crf=crf)
    except BaseException:
        try:
            out_path.unlink(missing_ok=True)
        except OSError:  # pragma: no cover - file already gone / locked
            pass
        raise

    elapsed = time.perf_counter() - started
    size = out_path.stat().st_size
    meta = {
        "name": name,
        "scene": scene_id,
        "path": path_name,
        "path_label": str(PATHS[path_name]["label"]),
        "seconds": round(written / float(fps), 3),
        "fps": fps,
        "width": width,
        "height": height,
        "frames": written,
        "light_from": light_from,
        "light_to": light_to,
        "wind": wind,
        "travel": round(min(DOLLY_MAX_M, max(0.0, travel)), 3),
        "points": scene.count,
        "bytes": size,
        "elapsed": round(elapsed, 2),
        "created": created.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        sidecar_path(out_path).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:  # pragma: no cover - a missing sidecar is not fatal
        pass

    meta["url"] = f"/api/world/video-file/{name}"
    return meta
