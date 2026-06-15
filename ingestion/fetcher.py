from __future__ import annotations

import asyncio
import os
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from ingestion.filter import SKIP_DIRS
from ingestion.types import (
    AuthError,
    Err,
    FetchedRepo,
    FetchError,
    InvalidSourceError,
    NetworkError,
    Ok,
    RepoTooLargeError,
    Result,
)

DEFAULT_FILE_LIMIT = 5000

_GITHUB_HTTPS = re.compile(r"^https?://github\.com/([^/]+)/([^/.]+)(?:\.git)?/?$")
_GITHUB_SSH = re.compile(r"^git@github\.com:([^/]+)/([^/.]+)(?:\.git)?$")


def _classify(source: str) -> str:
    if _GITHUB_HTTPS.match(source) or _GITHUB_SSH.match(source):
        return "github"
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https") and source.lower().endswith(".zip"):
        return "zip"
    if Path(source).exists():
        return "local"
    return "unknown"


def _https_with_token(url: str, token: str) -> str:
    parsed = urlparse(url)
    return f"https://{token}@{parsed.netloc}{parsed.path}"


async def _clone_github(
    source: str, token: str | None, cache_dir: Path
) -> Result[FetchedRepo, FetchError]:
    try:
        import git  # type: ignore[import-not-found]
    except ImportError as e:
        return Err(NetworkError(source=source, reason=f"gitpython unavailable: {e}"))

    target = cache_dir / _repo_slug(source)
    if target.exists():
        try:
            repo = git.Repo(target)
            return Ok(
                FetchedRepo(
                    root=target,
                    source_url=source,
                    commit_sha=repo.head.commit.hexsha,
                    fetched_at=datetime.now(UTC),
                )
            )
        except Exception:
            pass

    cache_dir.mkdir(parents=True, exist_ok=True)
    clone_url = _https_with_token(source, token) if token and source.startswith("http") else source

    def _do_clone() -> Result[FetchedRepo, FetchError]:
        try:
            repo = git.Repo.clone_from(clone_url, target, depth=1)
            return Ok(
                FetchedRepo(
                    root=target,
                    source_url=source,
                    commit_sha=repo.head.commit.hexsha,
                    fetched_at=datetime.now(UTC),
                )
            )
        except git.GitCommandError as e:  # type: ignore[attr-defined]
            stderr = (getattr(e, "stderr", "") or "").lower()
            if "authentication" in stderr or "could not read" in stderr or "403" in stderr:
                return Err(AuthError(source=source, reason=str(e)))
            return Err(NetworkError(source=source, reason=str(e)))
        except Exception as e:
            return Err(NetworkError(source=source, reason=str(e)))

    return await asyncio.to_thread(_do_clone)


def _repo_slug(source: str) -> str:
    m = _GITHUB_HTTPS.match(source) or _GITHUB_SSH.match(source)
    if m:
        return f"{m.group(1)}__{m.group(2)}"
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", source)[:64]


def _extract_zip(source: str, cache_dir: Path) -> Result[FetchedRepo, FetchError]:
    target = cache_dir / _repo_slug(source)
    target.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(source) as zf:
            for name in zf.namelist():
                if name.startswith("/") or ".." in Path(name).parts:
                    return Err(
                        InvalidSourceError(source=source, reason=f"unsafe zip entry: {name}")
                    )
            zf.extractall(target)
    except zipfile.BadZipFile as e:
        return Err(InvalidSourceError(source=source, reason=f"bad zip: {e}"))
    except FileNotFoundError as e:
        return Err(InvalidSourceError(source=source, reason=str(e)))
    return Ok(
        FetchedRepo(
            root=target,
            source_url=source,
            commit_sha=None,
            fetched_at=datetime.now(UTC),
        )
    )


def _resolve_local(source: str) -> Result[FetchedRepo, FetchError]:
    path = Path(source).expanduser().resolve()
    if not path.exists():
        return Err(InvalidSourceError(source=source, reason="path does not exist"))
    if not path.is_dir():
        return Err(InvalidSourceError(source=source, reason="path is not a directory"))
    return Ok(
        FetchedRepo(
            root=path,
            source_url=str(path),
            commit_sha=None,
            fetched_at=datetime.now(UTC),
        )
    )


def _count_files(root: Path) -> int:
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        count += len(filenames)
        if count > DEFAULT_FILE_LIMIT * 2:
            break
    return count


async def fetch(
    source: str,
    *,
    token: str | None = None,
    cache_dir: Path,
    file_limit: int = DEFAULT_FILE_LIMIT,
) -> Result[FetchedRepo, FetchError]:
    """Resolve a source string (GitHub URL, zip URL/path, or local dir) into a FetchedRepo."""
    kind = _classify(source)
    auth_token = token or os.environ.get("GITHUB_TOKEN")

    if kind == "github":
        result = await _clone_github(source, auth_token, cache_dir)
    elif kind == "zip":
        result = await asyncio.to_thread(_extract_zip, source, cache_dir)
    elif kind == "local":
        result = _resolve_local(source)
    else:
        return Err(InvalidSourceError(source=source, reason="unrecognized source format"))

    if not result.is_ok():
        return result

    repo: FetchedRepo = result.unwrap()
    count = await asyncio.to_thread(_count_files, repo.root)
    if count > file_limit:
        return Err(RepoTooLargeError(source=source, file_count=count, limit=file_limit))
    return Ok(repo)
