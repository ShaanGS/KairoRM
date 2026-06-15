from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingestion.fetcher import fetch
from ingestion.types import AuthError, InvalidSourceError, RepoTooLargeError


@pytest.mark.asyncio
async def test_local_path_returns_fetched_repo(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("print('hi')\n")
    result = await fetch(str(tmp_path), cache_dir=tmp_path / "_cache")
    assert result.is_ok()
    repo = result.unwrap()
    assert repo.root == tmp_path.resolve()
    assert repo.commit_sha is None


@pytest.mark.asyncio
async def test_local_path_missing_is_invalid_source(tmp_path: Path) -> None:
    result = await fetch(str(tmp_path / "nope"), cache_dir=tmp_path / "_cache")
    assert not result.is_ok()
    assert isinstance(result.error, InvalidSourceError)


@pytest.mark.asyncio
async def test_zip_with_path_traversal_rejected(tmp_path: Path) -> None:
    bad_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("../escape.txt", "evil")
    # _classify only marks remote .zip URLs as 'zip'; for local zip path we test _extract_zip via
    # a synthetic source that classifies as unknown — so we directly exercise the safety check.
    from ingestion.fetcher import _extract_zip

    result = _extract_zip(str(bad_zip), tmp_path / "_cache")
    assert not result.is_ok()
    assert isinstance(result.error, InvalidSourceError)
    assert "unsafe" in result.error.reason


@pytest.mark.asyncio
async def test_github_clone_auth_failure(tmp_path: Path) -> None:
    fake_git = MagicMock()

    class FakeGitCommandError(Exception):
        def __init__(self, msg: str) -> None:
            super().__init__(msg)
            self.stderr = "fatal: Authentication failed"

    fake_git.GitCommandError = FakeGitCommandError
    fake_git.Repo.clone_from.side_effect = FakeGitCommandError("auth nope")

    with patch.dict("sys.modules", {"git": fake_git}):
        result = await fetch(
            "https://github.com/owner/private", cache_dir=tmp_path / "_cache"
        )
    assert not result.is_ok()
    assert isinstance(result.error, AuthError)


@pytest.mark.asyncio
async def test_repo_too_large(tmp_path: Path) -> None:
    for i in range(20):
        (tmp_path / f"f{i}.py").write_text("x\n")
    result = await fetch(str(tmp_path), cache_dir=tmp_path / "_cache", file_limit=5)
    assert not result.is_ok()
    assert isinstance(result.error, RepoTooLargeError)
    assert result.error.file_count > 5


@pytest.mark.asyncio
async def test_skip_dirs_excluded_from_count(tmp_path: Path) -> None:
    # A handful of real source files...
    for i in range(3):
        (tmp_path / f"f{i}.py").write_text("x\n")
    # ...plus a bloated virtualenv that must not count toward the limit.
    venv = tmp_path / ".venv"
    venv.mkdir()
    for i in range(50):
        (venv / f"dep{i}.py").write_text("y\n")
    result = await fetch(str(tmp_path), cache_dir=tmp_path / "_cache", file_limit=5)
    assert result.is_ok()


@pytest.mark.asyncio
async def test_unrecognized_source(tmp_path: Path) -> None:
    result = await fetch("ftp://nope.example.com/x", cache_dir=tmp_path / "_cache")
    assert not result.is_ok()
    assert isinstance(result.error, InvalidSourceError)
