"""T05 - consolidated, end-to-end validation of the whole reproduction.

Aggregates every earlier check into ONE report (logs/T05-verify.json):

  A. Runtime environment
     A1 venv python version == 3.12.x
     A2 torch importable, CUDA available, correct build (2.7.1+cu128)
     A3 nunchaku importable (Windows build)
     A4 da2 importable; utils3d importable
     A5 worldgen importable; pytorch3d SHIM importable + numerics
     A6 no linux_x86_64 packages leaked into the venv
  B. Models (delegates to verify_models.py)
     B1 LeoXie/WorldGen, B2 haodongli/DA-2, B3 svdq-int4, B4 FLUX.1-dev components
  C. Artifacts
     C1 outputs/splat.ply is a valid 3DGS PLY (binary_little_endian, >0 gaussians)
     C2 viser client build present (build/index.html) -> :8080 can serve
  D. C-drive budget (reads logs/c_drive_verify.json if present)

Usage:
    python D:\\mm\\scripts\\validate_artifacts.py [--splat D:\\mm\\outputs\\splat.ply]
Exit 0 = all PASS, 1 = some FAIL.
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import config as cfg  # noqa: E402

RESULTS: List[Tuple[str, bool, str]] = []


def rec(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""), flush=True)


# --------------------------------------------------------------------------- #
# A. runtime environment
# --------------------------------------------------------------------------- #
def check_env() -> None:
    ver = ".".join(map(str, sys.version_info[:3]))
    rec("A1 python 3.12.x", sys.version_info[:2] == (3, 12), f"python {ver}")

    try:
        import torch  # type: ignore

        cuda_ok = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if cuda_ok else "n/a"
        rec("A2 torch + CUDA", cuda_ok and torch.__version__.startswith("2.7.1"),
            f"torch {torch.__version__}, cuda {torch.version.cuda}, avail={cuda_ok}, gpu={name}")
    except Exception as exc:  # noqa: BLE001
        rec("A2 torch + CUDA", False, f"{exc}")

    try:
        import nunchaku  # type: ignore

        rec("A3 nunchaku import", True, f"{getattr(nunchaku, '__version__', '?')}")
    except Exception as exc:  # noqa: BLE001
        rec("A3 nunchaku import", False, f"{exc}")

    try:
        importlib.import_module("da2.model.spherevit")
        importlib.import_module("utils3d")
        rec("A4 da2 + utils3d import", True)
    except Exception as exc:  # noqa: BLE001
        rec("A4 da2 + utils3d import", False, f"{exc}")

    try:
        import worldgen  # type: ignore
        from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix  # type: ignore

        import numpy as np  # type: ignore
        from scipy.spatial.transform import Rotation as R  # type: ignore

        for _ in range(3):
            q_in = np.random.randn(4)
            q_in /= np.linalg.norm(q_in)
            M = R.from_quat(q_in, scalar_first=True).as_matrix()
            import torch  # type: ignore

            q_out = matrix_to_quaternion(torch.tensor(M, dtype=torch.float64)).numpy()
            err = float(np.min([np.linalg.norm(q_out - q_in), np.linalg.norm(q_out + q_in)]))
            assert err < 1e-5, f"shim numerics err={err}"
        rec("A5 worldgen + pytorch3d shim", True, "shim numerics < 1e-5")
    except Exception as exc:  # noqa: BLE001
        rec("A5 worldgen + pytorch3d shim", False, f"{exc}")

    try:
        out = subprocess.run([str(cfg.VENV_PY), "-m", "pip", "list", "--format=freeze"],
                             capture_output=True, text=True)
        bad = [ln for ln in out.stdout.splitlines() if "linux_x86_64" in ln]
        rec("A6 no linux_x86_64 packages", len(bad) == 0, f"offenders={bad}")
    except Exception as exc:  # noqa: BLE001
        rec("A6 no linux_x86_64 packages", False, f"{exc}")


# --------------------------------------------------------------------------- #
# B. models (delegate to verify_models.py)
# --------------------------------------------------------------------------- #
def check_models() -> None:
    try:
        import verify_models as vm  # type: ignore

        for name, fn in {
            "B1 LeoXie/WorldGen": vm.check_hf_lora,
            "B2 haodongli/DA-2": vm.check_hf_da2,
            "B3 svdq-int4": vm.check_hf_svdq,
            "B4 FLUX.1-dev components": vm.check_flux_components,
        }.items():
            ok, msgs = fn()
            rec(name, ok, msgs[-1] if msgs else "")
    except Exception as exc:  # noqa: BLE001
        rec("B models (delegated)", False, f"{exc}")


# --------------------------------------------------------------------------- #
# C. artifacts
# --------------------------------------------------------------------------- #
def check_splat(path: Path) -> None:
    if not path.exists():
        rec("C1 splat.ply exists", False, f"missing {path}")
        return
    size = path.stat().st_size
    if size < 1024:
        rec("C1 splat.ply valid 3DGS", False, f"too small: {size} B")
        return
    raw = path.read_bytes()
    end = raw.find(b"end_header")
    head = raw[: (end if end != -1 else 2048)]
    is_ply = raw.startswith(b"ply")
    fmt = re.search(rb"format (\w+)", head)
    m = re.search(rb"element vertex (\d+)", head)
    n = int(m.group(1)) if m else 0
    ok = is_ply and fmt is not None and fmt.group(1) == b"binary_little_endian" and n > 0
    rec("C1 splat.ply valid 3DGS", ok,
        f"{size:,} B, format={fmt.group(1).decode() if fmt else '?'}, gaussians={n:,}")

    if end == -1:
        rec("C1b splat numeric sample", False, "no end_header found")
        return
    try:
        import numpy as np  # type: ignore

        # header lists 'property float <name>' in vertex order -> build column offsets
        props = re.findall(rb"property\s+float\s+(\w+)", head)
        if not props:
            rec("C1b splat numeric sample", False, "no float properties in header")
            return
        stride = len(props) * 4
        names = [p.decode() for p in props]
        body_start = end + len(b"end_header")
        if raw[body_start:body_start + 1] == b"\n":
            body_start += 1

        def sample_at(v0: int, k: int) -> "np.ndarray":
            off = body_start + v0 * stride
            buf = raw[off: off + k * stride]
            arr = np.frombuffer(buf, dtype="<f4")
            return arr[: (len(arr) // stride) * stride].reshape(-1, len(props))

        k = 2000
        regions = [0, max(0, n // 2), max(0, n - k)]
        mats = [sample_at(v, k) for v in regions if v >= 0]
        samples = np.concatenate(mats, axis=0)

        overall_finite = float(np.isfinite(samples).mean())
        # position columns (x,y,z) must be real geometry, not NaN
        xyz_idx = [names.index(c) for c in ("x", "y", "z") if c in names]
        xyz = samples[:, xyz_idx]
        xyz_finite = float(np.isfinite(xyz).mean())
        xyz_nonzero = bool((xyz != 0).any())
        has_xyz = len(xyz_idx) == 3

        good = has_xyz and xyz_finite > 0.999 and xyz_nonzero
        rec("C1b splat numeric sample", good,
            f"sampled {samples.shape[0]} verts x {len(props)} props; "
            f"xyz_finite={xyz_finite:.4f}, xyz_nonzero={xyz_nonzero}, overall_finite={overall_finite:.4f}")
    except Exception as exc:  # noqa: BLE001
        rec("C1b splat numeric sample", False, f"{exc}")


def check_viser_client() -> None:
    build = cfg.REPO_DIR / "submodules" / "viser" / "src" / "viser" / "client" / "build" / "index.html"
    try:
        import viser  # type: ignore
        have_mod = True
    except Exception:  # noqa: BLE001
        have_mod = False
    rec("C2 viser module + client build", have_mod and build.exists(),
        f"module={have_mod}, build/index.html={build.exists()}")


# --------------------------------------------------------------------------- #
# D. C-drive budget report
# --------------------------------------------------------------------------- #
def report_c_drive() -> None:
    p = cfg.LOG_DIR / "c_drive_verify.json"
    if not p.exists():
        print("[INFO] c_drive_verify.json absent (run verify_c_drive.ps1 first)", flush=True)
        return
    data = json.loads(p.read_text(encoding="utf-8-sig"))
    for t in data.get("targets", []):
        ok = t.get("within_budget", False)
        rec(f"D {t['path']}", ok, f"delta={t.get('delta_gb')} GB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splat", default=str(cfg.OUTPUT_DIR / "splat.ply"))
    args = ap.parse_args()

    print("=" * 60)
    print("T05 validate_artifacts")
    print("=" * 60)
    check_env()
    check_models()
    check_splat(Path(args.splat))
    check_viser_client()
    report_c_drive()

    all_ok = all(ok for _, ok, _ in RESULTS)
    report = {
        "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in RESULTS],
        "all_ok": all_ok,
    }
    out = cfg.LOG_DIR / "T05-verify.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    state = cfg.load_state()
    state.setdefault("verify", {})["T05"] = {"all_ok": all_ok, "n_checks": len(RESULTS)}
    cfg.save_state(state)

    print("-" * 60)
    print(f"T05 VALIDATE: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'} ({len(RESULTS)} checks)")
    print(f"report: {out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
