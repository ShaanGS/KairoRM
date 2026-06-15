"""Chunk embedding with a free-tier-first, always-falls-back strategy.

Primary backend is Gemini's `text-embedding-004` (free tier, batched in groups of
100). If the API key is missing or the call fails for any reason — quota, network,
auth — we transparently fall back to a fully local `sentence-transformers` model so
indexing never hard-fails on a missing credential. The active backend is announced
through rich so the user always knows whether they're spending free-tier quota or
running locally.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

from rich.console import Console

from ingestion.types import EmbeddedChunk, Err, Ok, RankedChunk, Result

GEMINI_MODEL = "models/text-embedding-004"
GEMINI_BATCH_SIZE = 100
LOCAL_MODEL = "all-MiniLM-L6-v2"

# Output to stderr so embedding logs never pollute machine-readable stdout.
console = Console(stderr=True)


@dataclass(frozen=True, slots=True)
class EmbedError:
    reason: str


def _embed_text(chunk: RankedChunk) -> str:
    """The exact string we embed: scope header + code body.

    The context header is what lets a retrieved chunk carry its own scope
    (`# python | class Auth > method verify`) into the embedding space, so a query
    about "auth verification" lands near it even when the body alone is terse.
    """
    return f"{chunk.chunk.context_header}\n{chunk.chunk.content}"


def _embed_gemini(texts: list[str], api_key: str) -> list[list[float]]:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    out: list[list[float]] = []
    for start in range(0, len(texts), GEMINI_BATCH_SIZE):
        batch = texts[start : start + GEMINI_BATCH_SIZE]
        resp = genai.embed_content(model=GEMINI_MODEL, content=batch)
        out.extend(resp["embedding"])
    return out


_local_model = None


def _embed_local(texts: list[str]) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer

    global _local_model
    if _local_model is None:
        _local_model = SentenceTransformer(LOCAL_MODEL)
    vectors = _local_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return [row.tolist() for row in vectors]


async def embed_texts(
    texts: list[str], *, api_key: str | None = None
) -> Result[list[list[float]], EmbedError]:
    """Embed raw strings, returning one vector per input (Gemini → local fallback)."""
    if not texts:
        return Ok([])

    key = api_key or os.environ.get("GEMINI_API_KEY")

    if key:
        try:
            vectors = await asyncio.to_thread(_embed_gemini, texts, key)
            console.log(f"[green]Embedded {len(texts)} texts via Gemini ({GEMINI_MODEL})[/]")
            return Ok(vectors)
        except Exception as exc:  # quota, auth, network — fall back rather than fail
            console.log(
                f"[yellow]Gemini embedding failed ({exc}); "
                f"falling back to local {LOCAL_MODEL}[/]"
            )

    try:
        vectors = await asyncio.to_thread(_embed_local, texts)
        console.log(f"[cyan]Embedded {len(texts)} texts via local {LOCAL_MODEL}[/]")
        return Ok(vectors)
    except Exception as exc:
        return Err(EmbedError(reason=f"local embedding failed: {exc}"))


async def embed(
    chunks: list[RankedChunk], *, api_key: str | None = None
) -> Result[list[EmbeddedChunk], EmbedError]:
    """Embed ranked chunks into `EmbeddedChunk`s, preserving rank/importance."""
    if not chunks:
        return Ok([])

    texts = [_embed_text(c) for c in chunks]
    result = await embed_texts(texts, api_key=api_key)
    if not result.is_ok():
        return result

    vectors = result.unwrap()
    return Ok(
        [
            EmbeddedChunk(ranked=chunk, embedding=vector)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
    )
