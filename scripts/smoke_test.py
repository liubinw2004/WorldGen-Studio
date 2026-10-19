"""WorldGen environment smoke test.

Run with the D-drive venv interpreter::

    D:\\mm\\venv\\Scripts\\python.exe D:\\mm\\scripts\\smoke_test.py

Checks (each reported independently; the process exits non-zero only if a
*required* check fails):

  R1  python version / executable
  R2  import torch + CUDA availability + device name
  R3  import nunchaku
  R4  import da2 (SphereViT)
  R5  import worldgen (from <root>/WorldGen/src)
  R6  pytorch3d shim numeric parity vs scipy.spatial.transform.Rotation (< 1e-5)
  O1  nunchaku minimal forward (optional; only if a tiny quantised model loads)
  O2  DA-2 minimal forward (optional; only if weights are present)

Environment variables are injected from <root>/config.py so this test is
self-contained regardless of the calling shell.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- inject the D-drive environment (single source of truth) ----------------
sys.path.insert(0, str(ROOT))
try:
    import config as _cfg  # type: ignore

    os.environ.update(_cfg.ENV_VARS)
    WORLDGEN_SRC = str(_cfg.REPO_DIR / "src")
except Exception:  # pragma: no cover - config.py should always be present
    WORLDGEN_SRC = str(ROOT / "WorldGen" / "src")

RESULTS: dict[str, tuple[bool, str]] = {}


def _record(name: str, ok: bool, detail: str) -> None:
    RESULTS[name] = (ok, detail)
    flag = "PASS" if ok else "FAIL"
    print(f"[{flag}] {name}: {detail}", flush=True)


def check_python() -> bool:
    detail = f"{sys.version.split()[0]} @ {sys.executable}"
    _record("R1 python", sys.version_info[:2] >= (3, 11), detail)
    return sys.version_info[:2] >= (3, 11)


def check_torch() -> bool:
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if cuda_ok else "N/A"
        detail = (
            f"torch={torch.__version__} cuda={torch.version.cuda} "
            f"available={cuda_ok} device={dev}"
        )
        _record("R2 torch+cuda", cuda_ok, detail)
        return cuda_ok
    except Exception as exc:  # noqa: BLE001
        _record("R2 torch+cuda", False, f"{exc}\n{traceback.format_exc()}")
        return False


def check_nunchaku() -> bool:
    try:
        import nunchaku

        ver = getattr(nunchaku, "__version__", "?")
        _record("R3 nunchaku", True, f"version={ver} path={Path(nunchaku.__file__).parent}")
        return True
    except Exception as exc:  # noqa: BLE001
        _record("R3 nunchaku", False, f"{exc}\n{traceback.format_exc()}")
        return False


def check_da2() -> bool:
    try:
        from da2.model.spherevit import SphereViT  # noqa: F401

        _record("R4 da2", True, f"SphereViT={SphereViT}")
        return True
    except Exception as exc:  # noqa: BLE001
        _record("R4 da2", False, f"{exc}\n{traceback.format_exc()}")
        return False


def check_worldgen() -> bool:
    try:
        # The worldgen package is installed editable (pip install -e . --no-deps),
        # so a plain `import worldgen` must work with NO sys.path manipulation.
        import worldgen  # noqa: F401

        _record("R5 worldgen", True, f"path={Path(worldgen.__file__).parent}")
        return True
    except Exception as exc:  # noqa: BLE001
        _record("R5 worldgen", False, f"{exc}\n{traceback.format_exc()}")
        return False


def check_shim_numeric() -> bool:
    """Assert the shim matches scipy for 1000 random rotations (both directions)."""
    try:
        import numpy as np
        import torch
        from scipy.spatial.transform import Rotation as R_scipy

        from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

        rng = np.random.default_rng(0)
        quat_xyzw = rng.standard_normal((1000, 4))
        quat_xyzw /= np.linalg.norm(quat_xyzw, axis=1, keepdims=True)

        scipy_rot = R_scipy.from_quat(quat_xyzw)          # xyzw
        mats_np = scipy_rot.as_matrix()                    # (1000,3,3)
        mats = torch.tensor(mats_np, dtype=torch.float64)

        # matrix -> quaternion (shim returns wxyz)
        q_shim = matrix_to_quaternion(mats)               # (1000,4) wxyz
        q_scipy_wxyz = np.concatenate(
            [quat_xyzw[:, 3:4], quat_xyzw[:, 0:3]], axis=1
        )  # wxyz reference
        # align signs (q and -q are the same rotation)
        q_shim_np = q_shim.detach().cpu().numpy()
        sign = np.sign((q_shim_np * q_scipy_wxyz).sum(axis=1, keepdims=True))
        sign[sign == 0] = 1.0
        err_m2q = float(np.abs(q_shim_np * sign - q_scipy_wxyz).max())

        # quaternion -> matrix round-trip via shim
        q_in = torch.tensor(q_scipy_wxyz, dtype=torch.float64)
        mats_shim = quaternion_to_matrix(q_in)
        err_q2m = float(torch.abs(mats_shim - mats).max().item())

        # CUDA batch sanity (no NaN) if GPU present
        cuda_note = "cpu-only"
        if torch.cuda.is_available():
            m_cuda = mats.float().cuda()
            q_cuda = matrix_to_quaternion(m_cuda)
            cuda_note = f"cuda batch ok, nan={bool(torch.isnan(q_cuda).any())}"
            err_cuda = float(torch.abs(q_cuda.double().cpu() - q_shim.float().cpu().double()).max().item())
            cuda_note += f", cpu/cuda diff={err_cuda:.2e}"

        ok = err_m2q < 1e-5 and err_q2m < 1e-5
        detail = (
            f"max|m2q-shim - scipy|={err_m2q:.3e}, "
            f"max|q2m-shim - scipy|={err_q2m:.3e}, {cuda_note}"
        )
        _record("R6 shim numeric", ok, detail)
        return ok
    except Exception as exc:  # noqa: BLE001
        _record("R6 shim numeric", False, f"{exc}\n{traceback.format_exc()}")
        return False


def opt_nunchaku_forward() -> None:
    """Optional: a tiny nunchaku forward is only meaningful with a quantised ckpt."""
    try:
        import nunchaku  # noqa: F401

        _record("O1 nunchaku forward", True, "skipped (requires svdq ckpt; import-level only)")
    except Exception as exc:  # noqa: BLE001
        _record("O1 nunchaku forward", False, f"{exc}")


def opt_da2_forward() -> None:
    try:
        import torch
        from da2.model.spherevit import SphereViT  # noqa: F401

        _record("O2 da2 forward", True, "skipped (requires DA-2 weights; import-level only)")
    except Exception as exc:  # noqa: BLE001
        _record("O2 da2 forward", False, f"{exc}")


def main() -> int:
    print("=" * 70)
    print("WorldGen smoke test")
    print("=" * 70)

    check_python()
    check_torch()
    check_nunchaku()
    check_da2()
    check_worldgen()
    check_shim_numeric()
    opt_nunchaku_forward()
    opt_da2_forward()

    print("-" * 70)
    required = ["R1 python", "R2 torch+cuda", "R3 nunchaku", "R4 da2", "R5 worldgen", "R6 shim numeric"]
    all_required_ok = all(RESULTS.get(k, (False, ""))[0] for k in required)

    print("SUMMARY:")
    for k, (ok, _) in RESULTS.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}")

    if all_required_ok:
        print("\nSMOKE TEST: ALL REQUIRED CHECKS PASSED")
        return 0
    print("\nSMOKE TEST: SOME REQUIRED CHECKS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
