from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from indexing.retriever import retrieve
from indexing.vectorstore import store
from ingestion.types import Chunk, EmbeddedChunk, Ok, RankedChunk

REPO_ID = "b" * 64


def _embedded(
    chunk_id: str,
    *,
    content: str,
    embedding: list[float],
    importance: float = 0.1,
) -> EmbeddedChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        file_path=Path(f"/repo/{chunk_id}.py"),
        language="python",
        unit_type="function",
        name=chunk_id,
        start_line=1,
        end_line=3,
        content=content,
        token_count=10,
        imports=(),
        calls=(),
        context_header=f"# python | function {chunk_id}",
    )
    return EmbeddedChunk(
        ranked=RankedChunk(chunk=chunk, importance_score=importance), embedding=embedding
    )


def _mock_query_vec(vec: list[float]):
    async def _fake_embed_texts(texts, *, api_key=None):  # noqa: ANN001
        return Ok([vec])

    return patch("indexing.embeddings.embed_texts", side_effect=_fake_embed_texts)


@pytest.mark.asyncio
async def test_keyword_match_appears_in_results(tmp_path: Path) -> None:
    chunks = [
        _embedded("authn", content="def authenticate(user): return verify_password(user)",
                  embedding=[0.0, 0.0, 1.0]),
        _embedded("math", content="def add(a, b): return a + b", embedding=[1.0, 0.0, 0.0]),
        _embedded("io", content="def read_file(path): return open(path).read()",
                  embedding=[0.0, 1.0, 0.0]),
    ]
    await store(chunks, repo_id=REPO_ID, db_path=tmp_path / "db")

    # Query embedding deliberately neutral; the keyword "authenticate" must carry it.
    with _mock_query_vec([0.0, 0.0, 0.0]):
        result = await retrieve("authenticate the user", repo_id=REPO_ID, db_path=tmp_path / "db")

    assert result.is_ok()
    names = [rc.chunk.chunk_id for rc in result.unwrap()]
    assert "authn" in names
    assert names[0] == "authn"  # strongest keyword hit ranks first


@pytest.mark.asyncio
async def test_semantic_only_match_is_retrieved(tmp_path: Path) -> None:
    # "secret" chunk shares NO tokens with the query, but its embedding matches it.
    chunks = [
        _embedded("secret", content="def plugh(): return xyzzy()", embedding=[0.0, 0.0, 1.0]),
        _embedded("decoy", content="def parse json config loader settings",
                  embedding=[1.0, 0.0, 0.0]),
    ]
    await store(chunks, repo_id=REPO_ID, db_path=tmp_path / "db")

    # Query vector identical to "secret" embedding → top semantic match,
    # even though the query words never appear in its content.
    with _mock_query_vec([0.0, 0.0, 1.0]):
        result = await retrieve(
            "database migration rollback", repo_id=REPO_ID, db_path=tmp_path / "db"
        )

    assert result.is_ok()
    names = [rc.chunk.chunk_id for rc in result.unwrap()]
    assert "secret" in names


@pytest.mark.asyncio
async def test_pagerank_boost_beats_stronger_semantic(tmp_path: Path) -> None:
    # high: central code, only moderate semantic match.
    # low:  perfect semantic match, but structurally unimportant.
    chunks = [
        _embedded("high", content="def core(): pass", embedding=[0.5, 0.5, 0.0], importance=0.9),
        _embedded("low", content="def leaf(): pass", embedding=[0.0, 0.0, 1.0], importance=0.02),
    ]
    await store(chunks, repo_id=REPO_ID, db_path=tmp_path / "db")

    # Query == "low" embedding → "low" wins semantic rank #1, "high" is #2.
    with _mock_query_vec([0.0, 0.0, 1.0]):
        result = await retrieve("anything", repo_id=REPO_ID, db_path=tmp_path / "db")

    assert result.is_ok()
    names = [rc.chunk.chunk_id for rc in result.unwrap()]
    # Despite weaker semantic match, the high-PageRank chunk is ranked first.
    assert names.index("high") < names.index("low")


@pytest.mark.asyncio
async def test_k_limits_result_count(tmp_path: Path) -> None:
    chunks = [
        _embedded(f"c{i}", content=f"def f{i}(): return {i}", embedding=[float(i), 1.0, 0.0])
        for i in range(8)
    ]
    await store(chunks, repo_id=REPO_ID, db_path=tmp_path / "db")

    with _mock_query_vec([1.0, 1.0, 0.0]):
        result = await retrieve("f", repo_id=REPO_ID, db_path=tmp_path / "db", k=5)

    assert result.is_ok()
    assert len(result.unwrap()) == 5


@pytest.mark.asyncio
async def test_empty_store_returns_empty(tmp_path: Path) -> None:
    # Nothing stored under this repo_id → load fails → RetrieveError surfaced.
    with _mock_query_vec([1.0, 0.0, 0.0]):
        result = await retrieve("anything", repo_id="c" * 64, db_path=tmp_path / "db")
    assert not result.is_ok()
