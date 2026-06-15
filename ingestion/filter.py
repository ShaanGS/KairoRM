from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator
from pathlib import Path

import pathspec

from ingestion.types import FetchedRepo, RawFile

SKIP_DIRS: frozenset[str] = frozenset(
    {
        "node_modules",
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "dist",
        "build",
        "target",
        ".next",
        ".nuxt",
        "vendor",
        "Pods",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)

SKIP_FILES: frozenset[str] = frozenset(
    {
        "package-lock.json",
        "yarn.lock",
        "poetry.lock",
        "Cargo.lock",
        "pnpm-lock.yaml",
        "uv.lock",
        "composer.lock",
        "Gemfile.lock",
    }
)

SKIP_SUFFIXES: tuple[str, ...] = (
    ".min.js",
    ".min.css",
    ".map",
    ".pb.go",
    "_pb2.py",
)

OVERSIZED_BYTES = 1_000_000
MINIFIED_BYTES = 50_000
MINIFIED_MAX_LINES = 5


def _load_gitignore(repo_root: Path) -> pathspec.PathSpec:
    patterns: list[str] = []
    gi = repo_root / ".gitignore"
    if gi.exists():
        patterns.extend(gi.read_text(errors="ignore").splitlines())
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def _looks_binary(head: bytes) -> bool:
    if b"\x00" in head:
        return True
    try:
        head.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


def _looks_minified(path: Path, size: int) -> bool:
    if size < MINIFIED_BYTES:
        return False
    try:
        with path.open("rb") as f:
            lines = 0
            for chunk in iter(lambda: f.read(8192), b""):
                lines += chunk.count(b"\n")
                if lines > MINIFIED_MAX_LINES:
                    return False
        return lines <= MINIFIED_MAX_LINES
    except OSError:
        return False


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _should_skip(rel: Path, spec: pathspec.PathSpec) -> bool:
    parts = rel.parts
    if any(p in SKIP_DIRS for p in parts):
        return True
    if rel.name in SKIP_FILES:
        return True
    name_lower = rel.name.lower()
    if any(name_lower.endswith(sfx) for sfx in SKIP_SUFFIXES):
        return True
    if spec.match_file(str(rel)):
        return True
    return False


def _inspect(path: Path, rel: Path) -> RawFile | None:
    try:
        size = path.stat().st_size
    except OSError:
        return None
    try:
        with path.open("rb") as f:
            head = f.read(512)
    except OSError:
        return None
    if _looks_binary(head):
        return None
    if _looks_minified(path, size):
        return None
    return RawFile(
        path=path,
        rel_path=rel,
        size_bytes=size,
        sha256=_sha256_of(path),
        oversized=size > OVERSIZED_BYTES,
    )


def _collect(repo: FetchedRepo) -> list[RawFile]:
    spec = _load_gitignore(repo.root)
    out: list[RawFile] = []
    for path in repo.root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(repo.root)
        if _should_skip(rel, spec):
            continue
        raw = _inspect(path, rel)
        if raw is not None:
            out.append(raw)
    return out


async def walk(repo: FetchedRepo) -> AsyncIterator[RawFile]:
    """Yield `RawFile` records for every source file under `repo.root` that survives filtering."""
    files = await asyncio.to_thread(_collect, repo)
    for f in files:
        yield f
