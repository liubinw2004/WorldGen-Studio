"""One-shot installer for WorldGen Studio (Python side).

``install.bat`` creates the venv and then hands over to this script, which does
everything that needs real logic: dependency installation, the WorldGen checkout
plus its two required submodules, the runtime shims, and a final verification.

Design notes
------------
* **Idempotent.** Every step is skipped when already satisfied, so re-running
  after a network hiccup continues where it stopped.
* **No models.** Model weights are deliberately NOT downloaded here -- they are
  large, partly gated, and the user is expected to place them. ``--models``
  prints the exact list; ``scripts/check_models.py`` verifies placement.
* **Explicit failure.** A step that fails stops the run with the command that
  failed and a hint, rather than leaving a half-broken venv.

Usage::

    python scripts/bootstrap.py            # full install
    python scripts/bootstrap.py --models   # only print the model checklist
    python scripts/bootstrap.py --verify   # only run the post-install checks
    python scripts/bootstrap.py --mirror   # use China mirrors for this run
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    print(f"[install] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[install] WARNING: {msg}", flush=True)


def die(msg: str, hint: str = "") -> None:
    print(f"\n[install] FAILED: {msg}", flush=True)
    if hint:
        print(f"[install] 提示: {hint}", flush=True)
    raise SystemExit(1)


def run(cmd: list[str], cwd: Path | None = None, check: bool = True,
        quiet: bool = False) -> subprocess.CompletedProcess:
    log("$ " + " ".join(cmd))
    proc = subprocess.run(
        cmd, cwd=str(cwd) if cwd else None,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.STDOUT if quiet else None,
        text=True,
    )
    if check and proc.returncode != 0:
        die(f"命令返回 {proc.returncode}: {' '.join(cmd)}")
    return proc


def pip(*args: str) -> subprocess.CompletedProcess:
    return run([sys.executable, "-m", "pip", *args])


def git(*args: str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run git, routing github.com through the mirror when one is configured."""
    cmd = ["git"]
    prefix = cfg.GITHUB_PROXIES[0] if cfg.GITHUB_PROXIES else ""
    if prefix:
        # Applies to the clone AND to submodule URLs resolved from .gitmodules.
        cmd += ["-c", f"url.{prefix}.insteadOf=https://github.com/"]
    return run(cmd + list(args), cwd=cwd, check=check)


def has_module(module: str, attr: str = "") -> bool:
    """True if the venv can import ``module`` (and optionally ``module.attr``)."""
    code = f"import {module}" + (f"; {module}.{attr}" if attr else "")
    proc = subprocess.run([sys.executable, "-c", code],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc.returncode == 0


def which_or_none(name: str) -> str | None:
    return shutil.which(name)


# --------------------------------------------------------------------------- #
# steps
# --------------------------------------------------------------------------- #
def step_env() -> None:
    log("生成 env/ 配置文件 ...")
    cfg.emit_all()


def step_torch() -> None:
    """torch + torchvision from the CUDA index.

    A plain `pip install torch` on Windows installs the CPU-only wheel, which
    silently makes every later step run at 1/50th speed. That is why this goes
    through --index-url explicitly.
    """
    if has_module("torch", "cuda"):
        proc = subprocess.run(
            [sys.executable, "-c",
             "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"],
            capture_output=True, text=True)
        log("torch 已安装: " + proc.stdout.strip())
        if "False" in proc.stdout:
            warn("torch 已安装但 CUDA 不可用。若显卡驱动正常，请重装 CUDA 版 torch。")
        return

    log(f"安装 torch {cfg.TORCH_VERSION} + torchvision {cfg.TORCHVISION_VERSION} (CUDA) ...")
    pip("install",
        f"torch=={cfg.TORCH_VERSION}", f"torchvision=={cfg.TORCHVISION_VERSION}",
        "--index-url", cfg.TORCH_INDEX_URL)
    if not has_module("torch", "cuda"):
        die("torch 安装后仍无法 import。",
            "确认 Python 版本为 3.11/3.12，并检查网络能否访问 download.pytorch.org")


def step_nunchaku() -> None:
    """The 4-bit FLUX transformer runtime.

    Windows wheels are not on PyPI at this version, so the release asset is
    installed directly. Without it, a 12 GB card cannot run FLUX.1-dev at all.
    """
    if has_module("nunchaku"):
        log("nunchaku 已安装")
        return

    url = cfg.proxied(cfg.NUNCHAKU_WHEEL_URL)
    log(f"安装 nunchaku {cfg.NUNCHAKU_VERSION} ...")
    proc = pip("install", url, check=False)
    if proc.returncode != 0:
        die(
            "nunchaku 安装失败。",
            f"可手动下载 {cfg.NUNCHAKU_WHEEL_URL} 后执行 "
            f"venv\\Scripts\\pip install <文件路径>",
        )
    if not has_module("nunchaku"):
        die("nunchaku 安装后无法 import。")


def step_worldgen_repo() -> None:
    """Clone ZiYang-xie/WorldGen plus the two submodules the pipeline needs.

    submodules/pytorch3d is replaced by vendor/pytorch3d_shim and
    submodules/ml-sharp is only used by --use_sharp, so both are skipped.
    """
    if (cfg.REPO_DIR / "demo.py").exists():
        log(f"WorldGen 源码已存在: {cfg.REPO_DIR}")
        return

    if not which_or_none("git"):
        die("未找到 git。", "请先安装 Git for Windows: https://git-scm.com/download/win")

    log(f"clone WorldGen -> {cfg.REPO_DIR}")
    git("clone", "--depth", "1", cfg.proxied(cfg.WORLDGEN_REPO), str(cfg.REPO_DIR))

    log("初始化子模块 viser / DA-2 ...")
    git("submodule", "update", "--init", "--depth", "1",
        "submodules/viser", "submodules/DA-2", cwd=cfg.REPO_DIR)

    if not (cfg.REPO_DIR / "src" / "worldgen" / "worldgen.py").exists():
        die("WorldGen 源码不完整。", f"删除 {cfg.REPO_DIR} 后重试。")


def step_worldgen_pkg() -> None:
    """Editable install of the worldgen package itself.

    --no-deps is mandatory: WorldGen/pyproject.toml pins a *linux_x86_64*
    nunchaku wheel, so a normal install fails on Windows. Its dependencies are
    installed by step_deps / step_torch / step_nunchaku instead.
    """
    if has_module("worldgen"):
        log("worldgen 包已安装")
        return
    log("安装 worldgen 包 (--no-deps) ...")
    pip("install", "-e", str(cfg.REPO_DIR), "--no-deps")
    if not has_module("worldgen"):
        die("worldgen 包安装后无法 import。")


def step_viser() -> None:
    """WorldGen ships a patched viser fork; upstream viser lacks its API."""
    if has_module("viser"):
        log("viser 已安装")
        return
    viser_dir = cfg.REPO_DIR / "submodules" / "viser"
    if not viser_dir.exists():
        die("缺少 submodules/viser。", "重新运行本脚本以初始化子模块。")
    log("安装 viser (WorldGen fork) ...")
    pip("install", "-e", str(viser_dir))
    if not has_module("viser"):
        die("viser 安装后无法 import。")


def step_da2() -> None:
    """DA-2 depth model package + its utils3d dependency.

    Both are installed with --no-deps: their declared dependencies pull a second
    torch build, which would clobber the CUDA wheel from step_torch.
    """
    if has_module("da2"):
        log("da2 已安装")
        return

    utils3d = "git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900"
    log("安装 utils3d (--no-deps) ...")
    pip("install", "--no-deps", cfg.proxied(utils3d))

    da2_dir = cfg.REPO_DIR / "submodules" / "DA-2" / "src"
    if da2_dir.exists():
        log("安装 da2 本地子模块 (--no-deps) ...")
        pip("install", "--no-deps", str(da2_dir))
    else:
        log("未找到 DA-2 子模块，改为从 GitHub 安装 ...")
        pip("install", "--no-deps", cfg.proxied("git+" + cfg.DA2_REPO_SUBDIR[len("git+"):]))

    if not has_module("da2", "model"):
        die("da2 安装后无法 import。")


def step_deps() -> None:
    log("安装运行依赖 ...")
    pip("install", "-r", str(cfg.REQUIREMENTS_WIN))


def step_pytorch3d_shim() -> None:
    """Register vendor/pytorch3d_shim on sys.path via a .pth file.

    WorldGen imports two pure-math helpers from pytorch3d; building the real
    package on Windows needs a full CUDA toolchain, so the shim provides exactly
    those two functions. A .pth file keeps it working without touching
    site-packages contents.
    """
    import sysconfig
    site = Path(sysconfig.get_paths()["purelib"])
    pth = site / "worldgen_pytorch3d_shim.pth"
    target = str(cfg.SHIM_DIR)
    if pth.exists() and pth.read_text(encoding="utf-8").strip() == target:
        log("pytorch3d shim 已注册")
        return
    pth.write_text(target + "\n", encoding="utf-8")
    log(f"pytorch3d shim 已注册 -> {pth}")


def step_patch() -> None:
    """Let models live in models/ instead of an HF-hub-shaped cache tree."""
    from importlib import util
    script = cfg.SCRIPTS_DIR / "patch_worldgen.py"
    spec = util.spec_from_file_location("patch_worldgen", script)
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    if mod.apply() != 0:
        die("WorldGen 补丁失败。")


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def verify() -> bool:
    """Import-level smoke test. Does not touch the GPU."""
    checks = [
        ("torch", "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"),
        ("nunchaku", "import nunchaku; print('nunchaku', nunchaku.__version__)"),
        ("worldgen", "import worldgen; print('worldgen OK')"),
        ("da2", "from da2.model.spherevit import SphereViT; print('da2 OK')"),
        ("viser", "import viser; print('viser', viser.__version__)"),
        ("pytorch3d shim", "from pytorch3d.transforms import matrix_to_quaternion; print('pytorch3d shim OK')"),
        ("flask", "import flask; print('flask', flask.__version__)"),
    ]
    ok = True
    print()
    for name, code in checks:
        proc = subprocess.run([sys.executable, "-c", code],
                              capture_output=True, text=True)
        if proc.returncode == 0:
            print(f"  [OK]   {name}: {proc.stdout.strip()}")
        else:
            ok = False
            tail = (proc.stderr or proc.stdout).strip().splitlines()
            print(f"  [FAIL] {name}: {tail[-1] if tail else 'import error'}")
    return ok


def report_models() -> None:
    """Print which model files are present and where the missing ones go."""
    print()
    print("  模型清单（需自行下载，详见 README 的「模型准备」一节）")
    print("  " + "-" * 70)

    flux = cfg.MODELS_DIR / "FLUX.1-dev"
    flux_ok = (flux / "model_index.json").exists()
    print(f"  [{'OK' if flux_ok else '缺失'}] FLUX.1-dev 文本编码器/VAE"
          f"  -> {flux}")

    fill = cfg.MODELS_DIR / "FLUX.1-Fill-dev"
    fill_ok = (fill / "model_index.json").exists()
    print(f"  [{'OK' if fill_ok else '缺失'}] FLUX.1-Fill-dev（仅图生场景需要）"
          f"  -> {fill}")

    hf_entries = {
        "LeoXie/WorldGen (LoRA)": "models--LeoXie--WorldGen",
        "haodongli/DA-2 (深度)": "models--haodongli--DA-2",
        "svdq-int4-flux.1-dev (4bit 主干)": "models--mit-han-lab--svdq-int4-flux.1-dev",
    }
    for label, dirname in hf_entries.items():
        p = cfg.HF_HUB_CACHE / dirname
        print(f"  [{'OK' if p.exists() else '缺失'}] {label}")

    sr_ok = any(cfg.SR_DIR.glob("*.pth")) if cfg.SR_DIR.exists() else False
    print(f"  [{'OK' if sr_ok else '可选'}] 超分权重 (4x-UltraSharp / Real-ESRGAN)"
          f"  -> {cfg.SR_DIR}")
    print()


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="WorldGen Studio installer")
    ap.add_argument("--models", action="store_true", help="only print the model checklist")
    ap.add_argument("--verify", action="store_true", help="only run post-install checks")
    ap.add_argument("--mirror", action="store_true", help="use China mirrors for this run")
    ap.add_argument("--skip-deps", action="store_true", help="skip the pip dependency step")
    args = ap.parse_args()

    if args.mirror:
        os.environ["WORLDGEN_MIRROR"] = "1"
        # config was imported before the flag was known; re-evaluate the URLs.
        import importlib
        importlib.reload(cfg)

    print("=" * 74)
    print("  WorldGen Studio 安装")
    print(f"  项目目录: {cfg.ROOT}")
    print(f"  Python  : {sys.executable}")
    print(f"  镜像模式: {'开' if cfg.USE_MIRROR else '关'}")
    print("=" * 74)

    if args.models:
        report_models()
        return 0

    if args.verify:
        ok = verify()
        report_models()
        return 0 if ok else 1

    if not (sys.prefix != sys.base_prefix):
        die("当前不在虚拟环境中。", "请通过 install.bat 运行，它会先创建 venv。")

    steps = [
        ("env", step_env),
        ("torch", step_torch),
        ("nunchaku", step_nunchaku),
        ("worldgen-repo", step_worldgen_repo),
        ("worldgen-pkg", step_worldgen_pkg),
        ("viser", step_viser),
        ("da2", step_da2),
        ("deps", None if args.skip_deps else step_deps),
        ("pytorch3d-shim", step_pytorch3d_shim),
        ("patch", step_patch),
    ]

    total = len([s for s in steps if s[1]])
    for i, (name, fn) in enumerate(steps, 1):
        if fn is None:
            continue
        print(f"\n--- [{i}/{total}] {name} " + "-" * max(0, 58 - len(name)))
        fn()

    ok = verify()
    report_models()

    print("=" * 74)
    if ok:
        print("  安装完成。接下来：")
        print("    1. 按上面的清单放好模型文件")
        print("    2. 双击 start.bat 启动")
        print("=" * 74)
        return 0

    print("  部分组件导入失败，请按上面的提示排查。")
    print("=" * 74)
    return 1


if __name__ == "__main__":
    sys.exit(main())
