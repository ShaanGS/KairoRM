from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from indexing import embeddings
from indexing.embeddings import embed
from ingestion.types import Chunk, RankedChunk


def _ranked(name: str, content: str = "def f(): pass", importance: float = 0.1) -> RankedChunk:
    chunk = Chunk(
        chunk_id=f"id_{name}",
        file_path=Path(f"/tmp/{name}.py"),
        language="python",
        unit_type="function",
        name=name,
        start_line=1,
        end_line=2,
        content=content,
        token_count=5,
        imports=(),
        calls=(),
        context_header=f"# python | function {name}",
    )
    return RankedChunk(chunk=chunk, importance_score=importance)


@pytest.mark.asyncio
async def test_gemini_key_present_returns_correct_embedding_length() -> None:
    chunks = [_ranked("a"), _ranked("b")]

    def fake_embed_content(model, content):  # noqa: ANN001
        return {"embedding": [[0.0] * 768 for _ in content]}

    with (
        patch("google.generativeai.configure") as cfg,
        patch("google.generativeai.embed_content", side_effect=fake_embed_content),
    ):
        result = await embed(chunks, api_key="fake-key")

    assert result.is_ok()
    embedded = result.unwrap()
    assert len(embedded) == 2
    assert all(len(ec.embedding) == 768 for ec in embedded)
    # rank/importance preserved through embedding
    assert embedded[0].ranked.importance_score == 0.1
    cfg.assert_called_once_with(api_key="fake-key")


@pytest.mark.asyncio
async def test_missing_key_falls_back_to_local(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    chunks = [_ranked("a"), _ranked("b")]

    def fake_local(texts):  # noqa: ANN001
        return [[1.0, 2.0, 3.0] for _ in texts]

    with (
        patch("google.generativeai.embed_content") as gem,
        patch("indexing.embeddings._embed_local", side_effect=fake_local),
    ):
        result = await embed(chunks)  # no key anywhere

    assert result.is_ok()
    embedded = result.unwrap()
    assert len(embedded) == 2
    assert embedded[0].embedding == [1.0, 2.0, 3.0]
    gem.assert_not_called()  # never touched Gemini without a key


@pytest.mark.asyncio
async def test_gemini_quota_error_falls_back_to_local() -> None:
    chunks = [_ranked("a")]

    def boom(texts, api_key):  # noqa: ANN001
        raise RuntimeError("429 Resource has been exhausted (quota)")

    def fake_local(texts):  # noqa: ANN001
        return [[9.0, 9.0] for _ in texts]

    with (
        patch("indexing.embeddings._embed_gemini", side_effect=boom),
        patch("indexing.embeddings._embed_local", side_effect=fake_local),
    ):
        result = await embed(chunks, api_key="fake-key")

    assert result.is_ok()
    assert result.unwrap()[0].embedding == [9.0, 9.0]


@pytest.mark.asyncio
async def test_batching_250_chunks_makes_3_gemini_calls() -> None:
    chunks = [_ranked(f"c{i}") for i in range(250)]

    def fake_embed_content(model, content):  # noqa: ANN001
        return {"embedding": [[0.0] * 8 for _ in content]}

    with (
        patch("google.generativeai.configure"),
        patch(
            "google.generativeai.embed_content", side_effect=fake_embed_content
        ) as mock_embed,
    ):
        result = await embed(chunks, api_key="fake-key")

    assert result.is_ok()
    assert len(result.unwrap()) == 250
    # 250 chunks / 100 per batch -> 3 API calls
    assert mock_embed.call_count == 3


@pytest.mark.asyncio
async def test_empty_input_returns_empty() -> None:
    result = await embed([])
    assert result.is_ok()
    assert result.unwrap() == []


def test_embed_text_includes_context_header() -> None:
    rc = _ranked("verify", content="return token")
    text = embeddings._embed_text(rc)
    assert text.startswith("# python | function verify")
    assert "return token" in text
