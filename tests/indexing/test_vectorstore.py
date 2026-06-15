from __future__ import annotations

from pathlib import Path

import pytest

from indexing.vectorstore import StoreError, load, store
from ingestion.types import Chunk, EmbeddedChunk, RankedChunk

REPO_ID = "a" * 64


def _embedded(i: int, *, importance: float = 0.1) -> EmbeddedChunk:
    chunk = Chunk(
        chunk_id=f"id_{i}",
        file_path=Path(f"/repo/mod{i}.py"),
        language="python",
        unit_type="function",
        name=f"func{i}",
        start_line=i,
        end_line=i + 5,
        content=f"def func{i}():\n    return {i}",
        token_count=7 + i,
        imports=("import os",),
        calls=(f"helper{i}",),
        context_header=f"# python | function func{i}",
    )
    ranked = RankedChunk(chunk=chunk, importance_score=importance)
    return EmbeddedChunk(ranked=ranked, embedding=[float(i), float(i) + 0.5, 0.25])


@pytest.mark.asyncio
async def test_store_then_load_roundtrip(tmp_path: Path) -> None:
    chunks = [_embedded(i, importance=0.01 * i) for i in range(10)]
    store_res = await store(chunks, repo_id=REPO_ID, db_path=tmp_path / "db")
    assert store_res.is_ok()

    load_res = await load(REPO_ID, tmp_path / "db")
    assert load_res.is_ok()
    loaded = load_res.unwrap()
    assert len(loaded) == 10

    by_id = {ec.ranked.chunk.chunk_id: ec for ec in loaded}
    original = _embedded(3, importance=0.03)
    got = by_id["id_3"]
    assert got.ranked.chunk.name == "func3"
    assert got.ranked.chunk.language == "python"
    assert got.ranked.chunk.unit_type == "function"
    assert got.ranked.chunk.start_line == 3
    assert got.ranked.chunk.token_count == original.ranked.chunk.token_count
    assert got.ranked.chunk.imports == ("import os",)
    assert got.ranked.chunk.calls == ("helper3",)
    assert got.ranked.chunk.content == original.ranked.chunk.content
    assert abs(got.ranked.importance_score - 0.03) < 1e-9
    assert got.embedding == [3.0, 3.5, 0.25]


@pytest.mark.asyncio
async def test_second_store_same_repo_is_skipped(tmp_path: Path) -> None:
    first = [_embedded(i) for i in range(10)]
    assert (await store(first, repo_id=REPO_ID, db_path=tmp_path / "db")).is_ok()

    # A different (smaller) payload under the same repo_id must be ignored.
    second = [_embedded(i) for i in range(100, 105)]
    assert (await store(second, repo_id=REPO_ID, db_path=tmp_path / "db")).is_ok()

    loaded = (await load(REPO_ID, tmp_path / "db")).unwrap()
    assert len(loaded) == 10  # still the original 10, not overwritten or appended
    ids = {ec.ranked.chunk.chunk_id for ec in loaded}
    assert ids == {f"id_{i}" for i in range(10)}


@pytest.mark.asyncio
async def test_invalid_db_path_returns_store_error(tmp_path: Path) -> None:
    # Make a regular file, then try to use a path *inside* it as the db dir.
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("i am a file")
    bad_db = blocker / "subdir"

    result = await store([_embedded(0)], repo_id=REPO_ID, db_path=bad_db)
    assert not result.is_ok()
    assert isinstance(result.error, StoreError)


@pytest.mark.asyncio
async def test_load_missing_collection_returns_error(tmp_path: Path) -> None:
    result = await load("f" * 64, tmp_path / "empty_db")
    assert not result.is_ok()
    assert isinstance(result.error, StoreError)


@pytest.mark.asyncio
async def test_store_empty_is_noop(tmp_path: Path) -> None:
    result = await store([], repo_id=REPO_ID, db_path=tmp_path / "db")
    assert result.is_ok()
