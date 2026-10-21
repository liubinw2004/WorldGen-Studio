"""Fetch GitHub sources without git.

``install.bat`` only needs Python, so requiring Git for Windows purely to pull
three public repositories is a bad first run: lots of people grab a ZIP from
GitHub and have no git installed at all. This module fetches the same trees as
``git clone`` + ``git submodule update`` through the archive endpoint:

    https://github.com/<owner>/<repo>/archive/refs/heads/<ref>.zip

The API is deliberately small -- :func:`fetch_repo` drops a flattened tree at a
destination directory, which is all ``bootstrap.py`` needs.

Note on TLS: on the machine this was developed on, libcurl's schannel backend
fails Windows' certificate-revocation check while Python's OpenSSL stack is
happy, so the verified attempt is tried first and an unverified context is used
only as a fallback.
"""

from __future__ import annotations

import os
import shutil
import ssl
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

USER_AGENT = "WorldGen-Studio-installer/1.0"
BRANCH_CANDIDATES = ("main", "master")


def _log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def _temp_zip(dest: Path) -> Path:
    """A scratch file inside the project tree -- never the system temp dir.

    Two reasons: the project keeps every write off the system drive, and
    ``mkstemp`` leaves an open handle that makes the file undeletable on
    Windows unless it is closed immediately.
    """
    dest.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(suffix=".zip.part", dir=str(dest))
    os.close(fd)
    return Path(name)


def _mirror_prefixes() -> list[str]:
    """Reuse whatever mirror config.py negotiated (``install.bat --mirror``)."""
    try:
        import config as cfg
        return [p for p in cfg.GITHUB_PROXIES if p]
    except Exception:  # noqa: BLE001
        return []


def _open(url: str, timeout: int = 60):
    """urlopen with a verified context, falling back to unverified."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.URLError as exc:
        reason = str(exc.reason) if hasattr(exc, "reason") else str(exc)
        if "CERTIFICATE" not in reason.upper() and "SSL" not in reason.upper():
            raise
        ctx = ssl._create_unverified_context()  # noqa: S323 - see module docstring
        return urllib.request.urlopen(req, timeout=timeout, context=ctx)


def _download(url: str, dest: Path) -> None:
    with _open(url) as resp, open(dest, "wb") as fh:
        shutil.copyfileobj(resp, fh, length=1 << 20)


def _archive_urls(slug: str, ref: str) -> list[str]:
    direct = f"https://github.com/{slug}/archive/refs/heads/{ref}.zip"
    urls = [direct]
    for prefix in _mirror_prefixes():
        urls.append(prefix.rstrip("/") + "/" + direct)
    return urls


def _extract_flat(zip_path: Path, dest: Path) -> None:
    """Unzip into dest, dropping GitHub's single top-level directory."""
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(dest.parent)) as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_path)
        entries = [p for p in tmp_path.iterdir() if p.is_dir()]
        if len(entries) == 1:
            src = entries[0]
        elif len(entries) == 0:
            raise RuntimeError("压缩包里没有目录")
        else:
            src = tmp_path
        for child in src.iterdir():
            target = dest / child.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(child), str(target))


def fetch_repo(slug: str, dest: Path, ref: str | None = None) -> None:
    """Place ``owner/repo`` at ``dest`` using the ZIP archive (no git needed).

    ``slug`` is ``"ZiYang-xie/WorldGen"``. When ``ref`` is None the branch name
    is probed (main, then master).
    """
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        _log(f"已存在，跳过: {dest}")
        return

    refs = [ref] if ref else list(BRANCH_CANDIDATES)
    last_error: Exception | None = None

    for candidate in refs:
        for url in _archive_urls(slug, candidate):
            tmp_zip = _temp_zip(dest.parent)
            try:
                _log(f"下载 {slug} ({candidate}) ...")
                _download(url, tmp_zip)
                _extract_flat(tmp_zip, dest)
                _log(f"完成: {dest}")
                return
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code == 404:
                    break  # branch name wrong, try the next ref
                continue
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
            finally:
                tmp_zip.unlink(missing_ok=True)

    raise RuntimeError(f"无法下载 {slug}: {last_error}")


def fetch_commit(slug: str, commit: str, dest: Path) -> None:
    """Same as fetch_repo but pinned to a commit sha (archive/refs/tags-less)."""
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        _log(f"已存在，跳过: {dest}")
        return
    direct = f"https://github.com/{slug}/archive/{commit}.zip"
    urls = [direct] + [p.rstrip("/") + "/" + direct for p in _mirror_prefixes()]
    last_error: Exception | None = None
    for url in urls:
        tmp_zip = _temp_zip(dest.parent)
        try:
            _log(f"下载 {slug}@{commit[:7]} ...")
            _download(url, tmp_zip)
            _extract_flat(tmp_zip, dest)
            _log(f"完成: {dest}")
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
        finally:
            tmp_zip.unlink(missing_ok=True)
    raise RuntimeError(f"无法下载 {slug}@{commit[:7]}: {last_error}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: fetch_source.py <owner/repo> <dest> [ref]")
        raise SystemExit(2)
    fetch_repo(sys.argv[1], Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None)
