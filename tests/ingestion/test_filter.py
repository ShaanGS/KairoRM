from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from ingestion.filter import walk
from ingestion.types import FetchedRepo


def _repo(root: Path) -> FetchedRepo:
    return FetchedRepo(
        root=root, source_url=str(root), commit_sha=None, fetched_at=datetime.now(UTC)
    )


async def _collect(repo: FetchedRepo) -> list[str]:
    return [str(f.rel_path) async for f in walk(repo)]


@pytest.mark.asyncio
async def test_gitignore_honored(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("secret.py\n*.log\n")
    (tmp_path / "main.py").write_text("print('hi')\n")
    (tmp_path / "secret.py").write_text("API=1\n")
    (tmp_path / "run.log").write_text("ok\n")
    paths = await _collect(_repo(tmp_path))
    assert "main.py" in paths
    assert "secret.py" not in paths
    assert "run.log" not in paths


@pytest.mark.asyncio
async def test_node_modules_and_dist_always_skipped(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("ok\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.js").write_text("ok\n")
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "bundle.js").write_text("ok\n")
    paths = await _collect(_repo(tmp_path))
    assert paths == ["main.py"]


@pytest.mark.asyncio
async def test_lock_files_skipped(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("ok\n")
    (tmp_path / "package-lock.json").write_text("{}\n")
    (tmp_path / "uv.lock").write_text("\n")
    paths = await _collect(_repo(tmp_path))
    assert paths == ["main.py"]


@pytest.mark.asyncio
async def test_misnamed_binary_skipped(tmp_path: Path) -> None:
    # .txt extension but actually PNG magic bytes + nulls
    (tmp_path / "fake.txt").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    (tmp_path / "real.py").write_text("ok\n")
    paths = await _collect(_repo(tmp_path))
    assert paths == ["real.py"]


@pytest.mark.asyncio
async def test_minified_skipped(tmp_path: Path) -> None:
    big_one_line = "a" * 200_000
    (tmp_path / "vendor.min.js").write_text(big_one_line)  # also matches .min.js suffix
    (tmp_path / "huge_one_line.js").write_text(big_one_line)  # caught by entropy heuristic
    (tmp_path / "ok.py").write_text("x\n")
    paths = await _collect(_repo(tmp_path))
    assert paths == ["ok.py"]


@pytest.mark.asyncio
async def test_oversized_emitted_but_flagged(tmp_path: Path) -> None:
    big = "line\n" * 300_000  # ~1.5 MB, many lines so not minified
    (tmp_path / "big.py").write_text(big)
    files = [f async for f in walk(_repo(tmp_path))]
    assert len(files) == 1
    assert files[0].oversized is True
