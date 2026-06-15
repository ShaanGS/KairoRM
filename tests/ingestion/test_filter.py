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
async def test_oversized_files_skipped(tmp_path: Path) -> None:
    # Files over the 1MB cap are skipped before being read — a single huge file must
    # not OOM/hang the run. (Previously emitted-but-flagged; the flag was never used.)
    (tmp_path / "big.py").write_text("line\n" * 300_000)  # ~1.5 MB
    (tmp_path / "small.py").write_text("x = 1\n")
    paths = await _collect(_repo(tmp_path))
    assert "small.py" in paths
    assert "big.py" not in paths


@pytest.mark.asyncio
async def test_symlinks_are_skipped(tmp_path: Path) -> None:
    # A hostile repo could symlink to host files to exfiltrate them; symlinks must be
    # skipped entirely so their (off-repo) contents never reach the index or the LLM.
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "host_secret.txt").write_text("SECRET")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "real.py").write_text("def f(): pass\n")
    import os

    os.symlink(outside / "host_secret.txt", repo / "leaked.txt")
    paths = await _collect(_repo(repo))
    assert "real.py" in paths
    assert "leaked.txt" not in paths
