"""WorldGen Studio - single source of truth for paths, env vars, mirrors, models.

This module is the ONLY place that knows where the project lives.

Portability
-----------
``ROOT`` is the directory containing this file, so the project runs from any
path (``D:/mm``, ``C:/Users/me/WorldGen``, ``/home/me/worldgen``, ...) with no
edits. Override with the ``WORLDGEN_ROOT`` environment variable when the data
(caches, models, outputs) should live somewhere other than the checkout:

    set WORLDGEN_ROOT=E:\\ai\\worldgen

Every other path in the project derives from ``ROOT``. Nothing outside this
file may hard-code an absolute project path.

Run directly to (re)generate the env/* helper scripts and the directory tree::

    python config.py --emit
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Dict, List

IS_WINDOWS = os.name == "nt"

# --------------------------------------------------------------------------- #
# Root discovery - the one and only anchor
# --------------------------------------------------------------------------- #
ROOT: Path = Path(os.environ.get("WORLDGEN_ROOT") or Path(__file__).resolve().parent)

VENV_DIR: Path = ROOT / "venv"
CACHE_DIR: Path = ROOT / "cache"
TMP_DIR: Path = ROOT / "tmp"
OUTPUT_DIR: Path = ROOT / "outputs"
VIDEO_DIR: Path = OUTPUT_DIR / "videos"     # rendered 360 panorama clips (mp4)
LOG_DIR: Path = ROOT / "logs"
MODELS_DIR: Path = ROOT / "models"
FLUX_DEV_DIR: Path = MODELS_DIR / "FLUX.1-dev"        # assembled diffusers components
FLUX_FILL_DIR: Path = MODELS_DIR / "FLUX.1-Fill-dev"  # assembled i2s components (no transformer/)
SR_DIR: Path = MODELS_DIR / "sr"                      # 4x-UltraSharp / Real-ESRGAN weights
VENDOR_DIR: Path = ROOT / "vendor"
SCRIPTS_DIR: Path = ROOT / "scripts"
ENV_DIR: Path = ROOT / "env"
DOCS_DIR: Path = ROOT / "docs"
REPO_DIR: Path = ROOT / "WorldGen"          # cloned ZiYang-xie/WorldGen
APP_DIR: Path = ROOT / "app"
STATIC_DIR: Path = APP_DIR / "static"
TOOLS_DIR: Path = ROOT / "tools"
WHEELS_DIR: Path = ROOT / "wheels"           # optional local .whl cache

# Shim / state
SHIM_DIR: Path = VENDOR_DIR / "pytorch3d_shim"
STATE_FILE: Path = ROOT / "state.json"

# Caches fully redirected under ROOT
HF_CACHE: Path = CACHE_DIR / "hf"
HF_HUB_CACHE: Path = HF_CACHE / "hub"
HF_TRANSFORMERS_CACHE: Path = HF_CACHE / "transformers"
TORCH_CACHE: Path = CACHE_DIR / "torch"
TORCH_INDUCTOR_CACHE: Path = TORCH_CACHE / "inductor"
PIP_CACHE: Path = CACHE_DIR / "pip"
UV_CACHE: Path = CACHE_DIR / "uv"
XDG_CACHE: Path = CACHE_DIR / "xdg"
MPL_CACHE: Path = CACHE_DIR / "matplotlib"
TRITON_CACHE: Path = CACHE_DIR / "triton"
CONDA_PKGS_CACHE: Path = CACHE_DIR / "conda_pkgs"
MODELSCOPE_CACHE: Path = CACHE_DIR / "modelscope"
UV_PYTHON_INSTALL_DIR: Path = TOOLS_DIR / "python"

PIP_INI: Path = ENV_DIR / "pip.ini"
GITCONFIG: Path = ENV_DIR / "gitconfig"
REQUIREMENTS_WIN: Path = ENV_DIR / "requirements.win.txt"


# --------------------------------------------------------------------------- #
# Interpreter discovery (no hard-coded user paths)
# --------------------------------------------------------------------------- #
def find_system_python() -> str:
    """Locate a CPython 3.11/3.12 interpreter to build the venv from.

    Order: ``WORLDGEN_PYTHON`` -> the Windows ``py`` launcher (exact versions
    first) -> the interpreter running this script -> ``python`` on PATH.

    Version matters: torch's cu128 wheels and nunchaku's Windows wheel are built
    for cp312, so a bare 3.13 install cannot run the pipeline.
    """
    env = os.environ.get("WORLDGEN_PYTHON")
    if env and Path(env).exists():
        return env

    if IS_WINDOWS:
        import subprocess
        for ver in ("-3.12", "-3.11"):
            try:
                out = subprocess.run(
                    ["py", ver, "-c", "import sys;print(sys.executable)"],
                    capture_output=True, text=True, timeout=15,
                )
                if out.returncode == 0 and out.stdout.strip():
                    return out.stdout.strip()
            except Exception:  # noqa: BLE001
                continue

    exe = Path(sys.executable)
    if exe.exists() and "venv" not in str(exe).lower() and ".venv" not in str(exe).lower():
        return str(exe)

    for name in (("python.exe", "python3.exe") if IS_WINDOWS else ("python3", "python")):
        found = shutil.which(name)
        if found:
            return found
    return ""


PY_EXE: str = find_system_python()
VENV_PY: Path = VENV_DIR / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
VENV_PIP: Path = VENV_DIR / ("Scripts/pip.exe" if IS_WINDOWS else "bin/pip")


# --------------------------------------------------------------------------- #
# Network - official sources by default, mirrors when opted in
# --------------------------------------------------------------------------- #
# Set WORLDGEN_MIRROR=1 (or pass --mirror to install.bat) on mainland-China
# networks: huggingface.co and github.com are frequently unreachable there.
USE_MIRROR: bool = os.environ.get("WORLDGEN_MIRROR", "").strip() not in ("", "0", "false")

HF_ENDPOINT: str = (
    "https://hf-mirror.com" if USE_MIRROR else "https://huggingface.co"
)

# GitHub proxies, ordered by preference. Only used when USE_MIRROR is on.
GITHUB_PROXIES: List[str] = [
    "https://ghfast.top/https://github.com/",
    "https://gh-proxy.com/https://github.com/",
] if USE_MIRROR else [""]

# PyPI mirrors, ordered by preference. Only used when USE_MIRROR is on.
PYPI_MIRRORS: List[str] = [
    "https://mirrors.cloud.tencent.com/pypi/simple/",
    "https://mirrors.aliyun.com/pypi/simple/",
] if USE_MIRROR else ["https://pypi.org/simple/"]

# torch CUDA wheels. cu128 matches the RTX 30/40/50 series and ships cp312
# win_amd64 builds; the default index keeps Windows users off the CPU-only wheel
# that plain `pip install torch` would otherwise pick.
TORCH_INDEX_URL: str = "https://download.pytorch.org/whl/cu128"
TORCH_VERSION: str = "2.7.1"
TORCHVISION_VERSION: str = "0.22.1"

# nunchaku Windows wheel (aligned to WorldGen's pyproject pin: v0.2.0).
NUNCHAKU_VERSION: str = "0.2.0"
NUNCHAKU_WHEEL_NAME: str = f"nunchaku-{NUNCHAKU_VERSION}+torch2.7-cp312-cp312-win_amd64.whl"
NUNCHAKU_WHEEL_URL: str = (
    "https://github.com/mit-han-lab/nunchaku/releases/download/"
    f"v{NUNCHAKU_VERSION}/" + NUNCHAKU_WHEEL_NAME
)

# Source repos.
WORLDGEN_REPO: str = "https://github.com/ZiYang-xie/WorldGen.git"
DA2_REPO_SUBDIR: str = "git+https://github.com/EnVision-Research/DA-2.git#subdirectory=src"
VISER_REPO: str = "git+https://github.com/ZiYang-xie/viser.git"


def proxied(url: str) -> str:
    """Route a GitHub URL through the configured mirror when enabled."""
    prefix = GITHUB_PROXIES[0] if GITHUB_PROXIES else ""
    if not prefix or not url.startswith("https://github.com/"):
        return url
    return prefix + url


# --------------------------------------------------------------------------- #
# ENV_VARS - every cache/temp path rebound under ROOT
# --------------------------------------------------------------------------- #
def _w(*parts: str) -> str:
    """Build a native-style path string under ROOT."""
    return str(ROOT.joinpath(*parts))


_NO_PROXY_HOSTS = (
    "hf-mirror.com,modelscope.cn,www.modelscope.cn,download.pytorch.org,"
    "mirrors.aliyun.com,mirrors.cloud.tencent.com,pypi.org,files.pythonhosted.org,"
    "github.com,ghfast.top,gh-proxy.com,ghproxy.net,codeload.github.com,"
    "huggingface.co,cdn.jsdelivr.net,localhost,127.0.0.1"
)

ENV_VARS: Dict[str, str] = {
    "HF_HOME": _w("cache", "hf"),
    "HUGGINGFACE_HUB_CACHE": _w("cache", "hf", "hub"),
    "HF_HUB_CACHE": _w("cache", "hf", "hub"),
    "TRANSFORMERS_CACHE": _w("cache", "hf", "transformers"),
    "TORCH_HOME": _w("cache", "torch"),
    "TORCHINDUCTOR_CACHE_DIR": _w("cache", "torch", "inductor"),
    "PIP_CACHE_DIR": _w("cache", "pip"),
    "PIP_CONFIG_FILE": _w("env", "pip.ini"),
    "UV_CACHE_DIR": _w("cache", "uv"),
    "UV_PYTHON_INSTALL_DIR": _w("tools", "python"),
    # TMPDIR wins over TMP/TEMP in tempfile.gettempdir(), so all three are set.
    "TMP": _w("tmp"),
    "TEMP": _w("tmp"),
    "TMPDIR": _w("tmp"),
    "XDG_CACHE_HOME": _w("cache", "xdg"),
    "MPLCONFIGDIR": _w("cache", "matplotlib"),
    "TRITON_CACHE_DIR": _w("cache", "triton"),
    "CONDA_PKGS_DIRS": _w("cache", "conda_pkgs"),
    "HF_ENDPOINT": HF_ENDPOINT,
    "GIT_CONFIG_GLOBAL": _w("env", "gitconfig"),
    "MODELSCOPE_CACHE": _w("cache", "modelscope"),
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    # The FLUX pipelines normally resolve the HF repo ids
    # "black-forest-labs/FLUX.1-dev"; scripts/patch_worldgen.py rewrites those two
    # literals to read these variables, which lets a plain diffusers download sit
    # in models/ instead of having to mimic the HF hub cache layout.
    "WORLDGEN_FLUX_DEV_DIR": _w("models", "FLUX.1-dev"),
    "WORLDGEN_FLUX_FILL_DIR": _w("models", "FLUX.1-Fill-dev"),
    # A host-level localhost proxy breaks CONNECT tunnels to the mirrors.
    "NO_PROXY": _NO_PROXY_HOSTS,
    "no_proxy": _NO_PROXY_HOSTS,
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
}

# Every directory that must exist before any install starts.
REQUIRED_DIRS: List[Path] = [
    CACHE_DIR, HF_CACHE, HF_HUB_CACHE, HF_TRANSFORMERS_CACHE,
    TORCH_CACHE, TORCH_INDUCTOR_CACHE, PIP_CACHE, UV_CACHE, XDG_CACHE,
    MPL_CACHE, TRITON_CACHE, CONDA_PKGS_CACHE, MODELSCOPE_CACHE,
    TMP_DIR, OUTPUT_DIR, VIDEO_DIR, LOG_DIR, MODELS_DIR, SR_DIR,
    VENDOR_DIR, SHIM_DIR, WHEELS_DIR, SCRIPTS_DIR, ENV_DIR, TOOLS_DIR,
]


# --------------------------------------------------------------------------- #
# Model manifest (what to download, how big, where it goes)
# --------------------------------------------------------------------------- #
MODELS: Dict[str, dict] = {
    "LeoXie/WorldGen": {
        "source": "huggingface",
        "gated": False,
        "approx_gb": 0.08,
        "tier": 1,
        "dest": "cache/hf/hub",
        "note": "Two LoRAs (t2s + i2s). Required.",
    },
    "haodongli/DA-2": {
        "source": "huggingface",
        "gated": False,
        "approx_gb": 1.28,
        "tier": 1,
        "dest": "cache/hf/hub",
        "note": "360-degree depth (SphereViT). Required.",
    },
    "mit-han-lab/svdq-int4-flux.1-dev": {
        "source": "huggingface",
        "gated": False,
        "approx_gb": 4.0,
        "tier": 1,
        "dest": "cache/hf/hub",
        "note": "4-bit nunchaku FLUX transformer. Required for <24GB VRAM.",
    },
    "black-forest-labs/FLUX.1-dev": {
        "source": "manual",
        "gated": True,
        "approx_gb": 33.7,
        "tier": 1,
        "dest": "models/FLUX.1-dev",
        "note": "GATED: needs a Hugging Face token and licence acceptance. "
                "Diffusers layout, text_encoder/vae/tokenizer/scheduler/model_index. "
                "The 23.8GB flux1-dev.safetensors is NOT needed (nunchaku supplies "
                "the transformer).",
    },
    "black-forest-labs/FLUX.1-Fill-dev": {
        "source": "manual",
        "gated": True,
        "approx_gb": 0.6,
        "tier": 2,
        "dest": "models/FLUX.1-Fill-dev",
        "note": "i2s mode only. Same layout as FLUX.1-dev plus the Fill transformer "
                "config; text encoders can be copied from FLUX.1-dev.",
    },
    "mit-han-lab/svdq-int4-flux.1-fill-dev": {
        "source": "huggingface",
        "gated": False,
        "approx_gb": 6.3,
        "tier": 2,
        "dest": "cache/hf/hub",
        "note": "4-bit nunchaku FLUX Fill transformer (i2s).",
    },
    "shi-labs/oneformer_ade20k_swin_large": {
        "source": "huggingface",
        "gated": False,
        "approx_gb": 0.9,
        "tier": 3,
        "dest": "cache/hf/hub",
        "note": "--inpaint_bg only.",
    },
    "comfyanonymous/flux_text_encoders": {
        "source": "huggingface",
        "gated": False,
        "approx_gb": 9.8,
        "tier": 3,
        "dest": "cache/hf/hub",
        "note": "Non-gated fallback text encoders (clip_l + t5xxl).",
    },
    "4x-UltraSharp": {
        "source": "manual",
        "gated": False,
        "approx_gb": 0.07,
        "tier": 2,
        "dest": "models/sr/4x-UltraSharp.pth",
        "note": "Optional: panorama super-resolution (sharp).",
    },
    "RealESRGAN_x4plus": {
        "source": "manual",
        "gated": False,
        "approx_gb": 0.07,
        "tier": 2,
        "dest": "models/sr/RealESRGAN_x4plus.pth",
        "note": "Optional: panorama super-resolution (softer, photo-like).",
    },
}

# Minimum free space (GiB) that must remain AFTER an operation's required space.
DISK_SAFETY_MARGIN_GB: int = 20
DISK_TARGET: str = str(ROOT)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def to_posix(win_path: str) -> str:
    """Convert ``D:\\mm\\x`` to ``/d/mm/x`` for Git Bash."""
    p = str(win_path).replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        p = "/" + p[0].lower() + p[2:]
    return p


def env_for_bash() -> Dict[str, str]:
    """Return ENV_VARS with values converted to MSYS/Git-Bash POSIX paths."""
    out: Dict[str, str] = {}
    for key, val in ENV_VARS.items():
        if len(val) > 1 and val[1] == ":":
            out[key] = to_posix(val)
        else:
            out[key] = val
    return out


def load_state() -> dict:
    """Load ``state.json`` (idempotent re-entry), returning an empty skeleton if missing."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"root": str(ROOT), "mirror": USE_MIRROR, "steps": {}}


def save_state(state: dict) -> None:
    """Persist state to ``<root>/state.json`` atomically."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE_FILE)


def mark_step(step: str, meta: dict | None = None) -> None:
    """Mark a step done in state.json (idempotent)."""
    from datetime import datetime, timezone

    state = load_state()
    state.setdefault("steps", {})[step] = {
        "done": True,
        "ts": datetime.now(timezone.utc).isoformat(),
        "meta": meta or {},
    }
    save_state(state)


def is_done(step: str) -> bool:
    """Return True if ``step`` was previously completed."""
    return bool(load_state().get("steps", {}).get(step, {}).get("done"))


def ensure_dirs() -> List[str]:
    """Create every REQUIRED_DIR and return the list created (as str)."""
    created: List[str] = []
    for d in REQUIRED_DIRS:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            created.append(str(d))
    return created


def inject_into_environ() -> None:
    """Inject all ENV_VARS into the current process (idempotent)."""
    os.environ.update(ENV_VARS)


# Apply immediately at import time so that *any* module that does
# `import config` has TMPDIR/HF_HOME/... in place BEFORE it imports torch or
# anything that caches tempfile.gettempdir().
os.environ.update(ENV_VARS)

# Some interpreters resolve `tempfile.tempdir` eagerly at import time; clearing
# the cache forces a re-resolve against the TMPDIR injected above.
try:  # pragma: no cover - interpreter-specific hardening
    import tempfile as _tempfile

    _tempfile.tempdir = None
except Exception:  # noqa: BLE001 - never break `import config` over this
    pass


# --------------------------------------------------------------------------- #
# Emitters (env/set_env.bat, env/env.sh, pip.ini, gitconfig, requirements)
# --------------------------------------------------------------------------- #
def _render_set_env_bat() -> str:
    lines = [
        "@echo off\r\n",
        "REM AUTO-GENERATED by config.py --emit. Do not edit by hand.\r\n",
        "REM Injects all env vars into the CURRENT cmd process only (no setx).\r\n",
    ]
    for key in sorted(ENV_VARS):
        lines.append(f'set "{key}={ENV_VARS[key]}"\r\n')
    lines.append("REM ensure directory tree exists\r\n")
    for d in REQUIRED_DIRS:
        lines.append(f'if not exist "{d}" mkdir "{d}"\r\n')
    return "".join(lines)


def _render_env_sh() -> str:
    bash_vars = env_for_bash()
    root_posix = to_posix(str(ROOT))
    lines = [
        "# AUTO-GENERATED by config.py --emit. Do not edit by hand.\n",
        f"# Usage:  source {root_posix}/env/env.sh\n",
        "# Git Bash does NOT inherit cmd 'set' vars -> this file must be sourced.\n",
        "\n",
    ]
    for key in sorted(bash_vars):
        lines.append(f"export {key}='{bash_vars[key]}'\n")
    lines.append("\n# ensure directory tree exists\n")
    for d in REQUIRED_DIRS:
        lines.append(f"mkdir -p '{to_posix(str(d))}'\n")
    return "".join(lines)


def _render_pip_ini() -> str:
    out = [
        "[global]\n",
        f"index-url = {PYPI_MIRRORS[0]}\n",
        f"cache-dir = {PIP_CACHE.as_posix()}\n",
        "timeout = 120\n",
        "retries = 5\n",
        "disable-pip-version-check = true\n",
        "no-cache-dir = false\n",
    ]
    if len(PYPI_MIRRORS) > 1:
        out.append(f"extra-index-url = {PYPI_MIRRORS[1]}\n")
    if USE_MIRROR:
        out.append("trusted-host = mirrors.aliyun.com\n")
        out.append("               mirrors.cloud.tencent.com\n")
    return "".join(out)


def _render_gitconfig(proxy: str) -> str:
    out = [
        "# AUTO-GENERATED by config.py --emit.\n",
        "# Global git config redirected here via GIT_CONFIG_GLOBAL so nothing\n",
        "# is written to the user's ~/.gitconfig.\n",
    ]
    if proxy:
        out += [
            f'[url "{proxy}"]\n    insteadOf = https://github.com/\n',
            f'[url "{proxy}"]\n    insteadOf = git@github.com:\n',
        ]
    out += ["[safe]\n    directory = *\n", "[core]\n    autocrlf = false\n"]
    return "".join(out)


def _render_activate_bat() -> str:
    root = str(ROOT)
    return (
        "@echo off\r\n"
        "REM AUTO-GENERATED by config.py --emit.\r\n"
        f'call "{root}\\env\\set_env.bat"\r\n'
        f'if exist "{root}\\venv\\Scripts\\activate.bat" (\r\n'
        f'    call "{root}\\venv\\Scripts\\activate.bat"\r\n'
        f'    echo [activate_worldgen] venv activated: {root}\\venv\r\n'
        ") else (\r\n"
        f'    echo [activate_worldgen] WARNING: venv not found at {root}\\venv\r\n'
        ")\r\n"
    )


def _render_requirements_win() -> str:
    """Explicit Windows dependency list.

    torch/torchvision/nunchaku are installed by install.bat from their own
    indexes (CUDA build + Windows-only wheel), so they are listed here only as
    comments -- a plain `pip install -r` would otherwise pull a CPU-only torch.
    """
    lines = [
        "# AUTO-GENERATED by config.py --emit. Do not edit by hand.\n",
        "# Install: python -m pip install -r env/requirements.win.txt\n",
        "#\n",
        "# torch / torchvision / nunchaku are installed separately by install.bat\n",
        "# (CUDA index + a Windows-only wheel) -- see the comments at the bottom.\n",
        "#\n",
        "# worldgen itself: pip install -e WorldGen --no-deps\n",
        "# (its pyproject.toml pins a linux_x86_64 nunchaku wheel, so --no-deps).\n",
        "\n",
    ]
    for spec in [
        "diffusers==0.33.1", "transformers==4.48.3", "peft>=0.7.1",
        "accelerate>=1.1.1", "safetensors", "sentencepiece", "einops",
        "timm>=1.0.15", "py360convert>=0.1.0", "pillow>=8.0.0",
        "opencv-python>=4.12.0.88", "open3d>=0.19.0", "trimesh>=4.6.1",
        "scikit-image>=0.24.0", "scipy", "imageio", "av>=13.0.0",
        "matplotlib", "tqdm",
        "plyfile", "huggingface_hub", "modelscope", "numpy",
        "flask>=3.0",
    ]:
        lines.append(f"{spec}\n")
    lines.append(
        f"\n# torch:     pip install torch=={TORCH_VERSION} torchvision=={TORCHVISION_VERSION}"
        f" --index-url {TORCH_INDEX_URL}\n"
    )
    lines.append(f"# nunchaku:  pip install {NUNCHAKU_WHEEL_URL}\n")
    lines.append(f"# viser:     pip install {VISER_REPO}\n")
    lines.append("# excluded: xformers (unused), pytorch3d (replaced by vendor/pytorch3d_shim)\n")
    return "".join(lines)


def emit_all(proxy: str | None = None) -> Dict[str, str]:
    """Write all generated env files. Returns {path: action}."""
    proxy = GITHUB_PROXIES[0] if proxy is None else proxy
    ensure_dirs()
    actions: Dict[str, str] = {}

    files = {
        ENV_DIR / "set_env.bat": _render_set_env_bat(),
        ENV_DIR / "env.sh": _render_env_sh(),
        PIP_INI: _render_pip_ini(),
        GITCONFIG: _render_gitconfig(proxy),
        REQUIREMENTS_WIN: _render_requirements_win(),
        ENV_DIR / "activate_worldgen.bat": _render_activate_bat(),
    }
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        existed = path.exists()
        path.write_text(content, encoding="utf-8", newline="")
        actions[str(path)] = "updated" if existed else "created"

    for d in (LOG_DIR, MODELS_DIR, OUTPUT_DIR, SR_DIR):
        keep = d / ".gitkeep"
        if not keep.exists():
            keep.write_text("", encoding="utf-8")

    if not STATE_FILE.exists():
        save_state(load_state())
        actions[str(STATE_FILE)] = "created"

    return actions


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WorldGen Studio config / path resolver.")
    parser.add_argument("--emit", action="store_true", help="generate env/*, pip.ini, gitconfig")
    parser.add_argument("--proxy", default=None, help="override the GitHub proxy prefix")
    parser.add_argument("--dirs", action="store_true", help="only create the directory tree")
    parser.add_argument("--env-json", action="store_true", help="print ENV_VARS as JSON")
    parser.add_argument("--show", action="store_true", help="print resolved paths")
    args = parser.parse_args(argv)

    if args.env_json:
        print(json.dumps(ENV_VARS, indent=2, ensure_ascii=False))
        return 0

    if args.show:
        print(f"root        : {ROOT}")
        print(f"mirror mode : {USE_MIRROR}")
        print(f"venv python : {VENV_PY} (exists={VENV_PY.exists()})")
        print(f"system py   : {PY_EXE or '(not found)'}")
        print(f"worldgen    : {REPO_DIR} (exists={REPO_DIR.exists()})")
        print(f"models      : {MODELS_DIR}")
        print(f"hf cache    : {HF_HUB_CACHE}")
        return 0

    if args.dirs:
        created = ensure_dirs()
        print(f"ensured {len(REQUIRED_DIRS)} dirs; newly created: {len(created)}")
        for c in created:
            print(f"  + {c}")
        return 0

    if args.emit:
        actions = emit_all(args.proxy)
        for path, action in actions.items():
            print(f"  {action}: {path}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
