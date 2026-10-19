"""Runtime memory shim for WorldGen's i2s (image-to-scene) panorama mapping.

Why this exists
---------------
WorldGen's shipped i2s path calls ``pred_depth`` which builds the per-pixel
"rays" with ``pano_unit_rays`` -> a FULL equirectangular grid (azimuth -pi..pi,
elevation -pi/2..pi/2), i.e. it treats the input *photo* as if it already covered
the whole sphere. ``map_image_to_pano`` then estimates the source FOV from those
rays (-> ~2*pi) and rescales to the target pano resolution, giving a source grid
of ~2048x1024 = ~2.1M rays. Finally ``batch_nearest_dot`` does

    sim = torch.mm(q_batch, src_dirs.t())     # (nn_batch, N_src)

with the default ``nn_batch=8192`` and ``N_src ~= 2.1e6`` -> a single
``(8192, 2.1e6)`` fp32 tensor = ~68 GB. This is impossible on any single GPU
(and on most CPU-RAM budgets), so the upstream i2s path OOMs regardless of the
``-r`` resolution, the VRAM-saving flags, or the input image.

What this shim does
-------------------
It replaces ONLY the query-chunk size used by ``batch_nearest_dot`` with a small
value so the *same* argmax is computed in many small chunks. The result is
mathematically identical to the upstream dense computation (argmax over the same
full similarity matrix), just streamed -- a pure memory optimisation. No WorldGen
source file is modified; the function object is rebound at runtime before
``demo.py`` imports ``worldgen``.

Usage::

    import sys
    sys.path.insert(0, "<project>/vendor/worldgen_i2s_shim")
    import worldgen_i2s_shim as shim
    shim.apply_patch(batch=64)          # or leave default
    # then run demo.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_QUERY_BATCH: int = 64

_PATCHED: bool = False
_LOGGED: bool = False


def _chunked_batch_nearest_dot(src_dirs: torch.Tensor,
                               query_dirs: torch.Tensor,
                               batch: int = DEFAULT_QUERY_BATCH) -> torch.Tensor:
    """Memory-bounded nearest-neighbour by max dot product.

    Identical result to the upstream ``batch_nearest_dot`` but streams the
    query dimension in ``batch``-sized chunks so the transient similarity matrix
    is ``(batch, N_src)`` instead of ``(8192, N_src)``.
    """
    global _LOGGED
    src = F.normalize(src_dirs, dim=1)
    query = F.normalize(query_dirs, dim=1)
    n_src = src.shape[0]
    n_q = query.shape[0]
    if not _LOGGED:
        transient_mib = (batch * n_src * 4) / (1024 ** 2)
        dense_mib = (8192 * n_src * 4) / (1024 ** 2)
        print(f"[i2s-shim] batch_nearest_dot: N_src={n_src:,} N_query={n_q:,} "
              f"chunk_batch={batch} (transient {transient_mib:.1f} MiB vs "
              f"upstream dense {dense_mib:.1f} MiB)", flush=True)
        _LOGGED = True
    out = []
    for start in range(0, n_q, batch):
        q = query[start:start + batch]              # (b, 3)
        sim = torch.mm(q, src.t())                  # (b, N_src)
        out.append(sim.argmax(dim=1))
    return torch.cat(out, dim=0)


def _patch_depth_offload() -> bool:
    """Move the DA-2 depth model off the GPU after its single i2s inference.

    In i2s the depth model runs once (``pred_depth`` on the input photo) and is
    not needed again until ``pred_pano_depth`` on the finished panorama. Leaving
    it resident (~1.3 GB) during the T5 prompt-encoding step drops free VRAM to
    ~8.6 GB -- still shy of the 9.5 GB T5-XXL, so ``accelerate`` spills part of
    the encoder and the step stalls. We relocate the depth model to CPU right
    after ``pred_depth`` and pull it back to GPU before ``pred_pano_depth``.
    """
    try:
        import worldgen.worldgen as wg
        from worldgen import pano_depth as pdmod
    except Exception as exc:  # noqa: BLE001
        print(f"[i2s-shim] could not import worldgen depth modules: {exc!r}", flush=True)
        return False

    def _mk_pred_depth(orig):
        def _pd(model, image):
            out = orig(model, image)
            try:
                model.to("cpu")
                torch.cuda.empty_cache()
                print("[i2s-shim] depth model -> CPU after pred_depth "
                      "(frees VRAM for T5 encode)", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[i2s-shim] depth->cpu failed: {exc!r}", flush=True)
            return out
        _pd._i2s_shim = True
        return _pd

    def _mk_pred_pano_depth(orig):
        def _ppd(model, image):
            try:
                model.to(torch.device("cuda"))
                torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001
                print(f"[i2s-shim] depth->cuda failed: {exc!r}", flush=True)
            return orig(model, image)
        _ppd._i2s_shim = True
        return _ppd

    for holder, name, mk in ((pdmod, "pred_depth", _mk_pred_depth),
                             (pdmod, "pred_pano_depth", _mk_pred_pano_depth),
                             (wg, "pred_depth", _mk_pred_depth),
                             (wg, "pred_pano_depth", _mk_pred_pano_depth)):
        if not hasattr(holder, name):
            continue
        cur = getattr(holder, name)
        if getattr(cur, "_i2s_shim", False):
            continue
        setattr(holder, name, mk(cur))
    print("[i2s-shim] patched pred_depth/pred_pano_depth (depth-model VRAM offload)",
          flush=True)
    return True


def _patch_t2s_depth_offload() -> bool:
    """In t2s the DA-2 depth model sits on the GPU from ``WorldGen.__init__``
    until ``pred_pano_depth`` runs -- i.e. it is resident for the ENTIRE
    diffusion step, even though t2s never calls ``pred_depth`` (that is the i2s
    photo path only).

    Measured on the 12 GB card: with it resident, the pre-encode free VRAM for
    t2s is 8,718 MiB, which is 800 MiB short of the 9.5 GB T5-XXL. accelerate
    then spills the encoder block-by-block and the run hangs silently at
    "Generating the world..." with the GPU pinned at 100%.

    Moving it to CPU for the duration of ``generate_world`` takes t2s to the
    same headroom i2s already has (~10 GB). ``pred_pano_depth`` is already
    wrapped by this shim and pulls the model back to CUDA when the panorama is
    ready, so nothing downstream changes.
    """
    try:
        import worldgen.worldgen as wg
    except Exception as exc:  # noqa: BLE001
        print(f"[i2s-shim] could not import worldgen.worldgen: {exc!r}", flush=True)
        return False

    orig = wg.WorldGen.generate_world
    if getattr(orig, "_i2s_shim", False):
        return True

    def _gw(self, *args, **kwargs):
        if getattr(self, "mode", "") == "t2s":
            dm = getattr(self, "depth_model", None)
            if dm is not None:
                try:
                    dm.to("cpu")
                    torch.cuda.empty_cache()
                    free = torch.cuda.mem_get_info()[0] / 1024 ** 2 if torch.cuda.is_available() else 0
                    print(f"[i2s-shim] t2s: depth model -> CPU before diffusion "
                          f"(GPU free={free:.0f} MiB)", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"[i2s-shim] t2s depth->cpu failed: {exc!r}", flush=True)
        return orig(self, *args, **kwargs)

    _gw._i2s_shim = True
    wg.WorldGen.generate_world = _gw
    print("[i2s-shim] patched WorldGen.generate_world (t2s depth-model VRAM offload)",
          flush=True)
    return True


def _patch_encode_prompt() -> bool:
    """Free GPU VRAM around the T5 prompt-encoding step.

    On a 12 GB card the resident nunchaku 4-bit transformer (~6.6 GB) + DA-2
    depth model (~1.3 GB) leave no room for the 9.5 GB T5-XXL. With
    ``enable_model_cpu_offload`` the T5 then gets swapped block-by-block to the
    CPU and the encode step stalls for ~20+ minutes (observed via py-spy:
    ``_get_t5_prompt_embeds -> _relative_position_bucket``).

    We therefore move the transformer off the GPU for the duration of
    ``encode_prompt`` and clear the CUDA cache; accelerate's cpu-offload hook
    reloads the transformer for the denoise loop, so the 50 diffusion steps still
    run on the GPU. Result of encoding is unchanged; only the memory placement
    differs.
    """
    # Two pipelines share the same FLUX encode_prompt shape: the Fill variant
    # (i2s) and the text-only variant (t2s). Both hold a 9.5 GB T5-XXL that
    # accelerate's lazy cpu-offload sequence never evicts once the transformer
    # has been moved aside manually -- so both need the same treatment.
    targets = []
    try:
        from worldgen.models.flux_pano_fill_pipeline import FluxFillPipeline
        targets.append(("FluxFillPipeline", FluxFillPipeline))
    except Exception as exc:  # noqa: BLE001
        print(f"[i2s-shim] could not import FluxFillPipeline: {exc!r}", flush=True)
    try:
        from worldgen.models.flux_pano_gen_pipeline import FluxPipeline as FluxGenPipeline
        targets.append(("FluxPipeline(t2s)", FluxGenPipeline))
    except Exception as exc:  # noqa: BLE001
        print(f"[i2s-shim] could not import gen FluxPipeline: {exc!r}", flush=True)
    if not targets:
        return False

    patched_any = False
    for cls_name, cls in targets:
        orig = cls.encode_prompt
        if getattr(orig, "_i2s_shim", False):
            patched_any = True
            continue
        cls.encode_prompt = _make_encode_wrapper(orig, cls_name)
        patched_any = True
        print(f"[i2s-shim] patched {cls_name}.encode_prompt (VRAM-free for T5)", flush=True)
    return patched_any


def _make_encode_wrapper(orig, cls_name: str):
    def _enc(self, *args, **kwargs):
        moved = False
        try:
            tr = getattr(self, "transformer", None)
            if tr is not None:
                tr.to("cpu")
                moved = True
        except Exception as exc:  # noqa: BLE001
            print(f"[i2s-shim] transformer.to(cpu) failed: {exc!r}", flush=True)
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        if torch.cuda.is_available():
            free = torch.cuda.mem_get_info()[0] / 1024 ** 2
            print(f"[i2s-shim] pre-encode: transformer offloaded={moved}, "
                  f"GPU free={free:.0f} MiB", flush=True)
        out = orig(self, *args, **kwargs)
        # CRITICAL: after encoding, evict BOTH text encoders from GPU.
        # enable_model_cpu_offload's model_cpu_offload_seq is
        # "text_encoder->text_encoder_2->transformer->vae" with LAZY eviction:
        # a component is only pushed off GPU when the *next* component comes up.
        # We manually moved the transformer off GPU before encode, so when the
        # VAE comes on stage its "predecessor" is already gone and nothing evicts
        # the 9.5 GB T5-XXL -> the VAE conv / randn allocations are squeezed into
        # ~125 MiB free, Windows WDDM pages instead of raising OOM, and the run
        # hangs silently forever. Explicitly moving the text encoders to CPU here
        # (frees ~9.9 GB) is what keeps the documented entry point stable.
        freed = []
        for name in ("text_encoder", "text_encoder_2"):
            enc = getattr(self, name, None)
            if enc is not None:
                try:
                    enc.to("cpu")
                    freed.append(name)
                except Exception as exc:  # noqa: BLE001
                    print(f"[i2s-shim] {name}.to(cpu) failed: {exc!r}", flush=True)
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        if torch.cuda.is_available():
            free = torch.cuda.mem_get_info()[0] / 1024 ** 2
            print(f"[i2s-shim] post-encode: text encoders -> CPU {freed}, "
                  f"GPU free={free:.0f} MiB", flush=True)
        return out

    _enc._i2s_shim = True
    return _enc


def apply_patch(batch: int = DEFAULT_QUERY_BATCH, free_vram_for_encode: bool = True) -> bool:
    """Apply the memory patches. Returns True if the core mapping patch succeeded."""
    global _PATCHED
    try:
        from worldgen.utils import general_utils as gu
    except Exception as exc:  # noqa: BLE001
        print(f"[i2s-shim] could not import worldgen.utils.general_utils: {exc!r}", flush=True)
        return False

    def _patched(src_dirs, query_dirs, batch_unused=8192):
        return _chunked_batch_nearest_dot(src_dirs, query_dirs, batch)

    gu.batch_nearest_dot = _patched
    _PATCHED = True
    print(f"[i2s-shim] patched worldgen.utils.general_utils.batch_nearest_dot "
          f"(query chunk = {batch})", flush=True)

    if free_vram_for_encode:
        _patch_depth_offload()
        _patch_t2s_depth_offload()
        _patch_encode_prompt()
    return True


def is_patched() -> bool:
    return _PATCHED
