"""Chunk embedding — local-first, with optional Gemini.

Embeddings run on a local `sentence-transformers` model by default: free, offline, no
rate limits, and the most reliable path on a fresh free-tier key (Gemini's embedding
quota is tiny and exhausts fast, which used to throw 429 noise and risk dimension
mismatches). Set `KAIRO_GEMINI_EMBED=1` to use Gemini embeddings instead. Whichever
backend is chosen first is pinned for the whole run so every vector shares one space.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import warnings
from dataclasses import dataclass

from ingestion.types import EmbeddedChunk, Err, Ok, RankedChunk, Result

GEMINI_MODEL = os.environ.get("KAIRO_EMBED_MODEL", "models/gemini-embedding-001")
GEMINI_BATCH_SIZE = 100
LOCAL_MODEL = "all-MiniLM-L6-v2"

# Quiet HuggingFace/tokenizers chatter before the local model ever loads.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Warnings/status go to the kairo logfile, never the terminal (kept clean for the TUI).
log = logging.getLogger("kairo")


@contextlib.contextmanager
def _silence_fd_stderr():
    """Redirect file descriptor 2 to /dev/null for the duration of the block.

    The HF Hub "unauthenticated requests" notice is printed straight to fd 2 by the
    hf_xet Rust extension, so Python's `warnings`/`logging` can't intercept it. We
    silence it only around the model load — exceptions still propagate, and this
    works whether or not the model is already cached (unlike HF_HUB_OFFLINE).
    """
    import sys

    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
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
        with warnings.catch_warnings(), _silence_fd_stderr():
            warnings.simplefilter("ignore")
            _local_model = SentenceTransformer(LOCAL_MODEL)
    vectors = _local_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return [row.tolist() for row in vectors]


# The active backend, pinned after the first embed of a run. Gemini (3072-d) and the
# local model (384-d) live in different vector spaces, so once we commit to one every
# later embed — including the retriever's per-query embeds — must use the same one, or
# vector-store lookups silently mismatch. Reset between tests via the autouse fixture.
_backend: str | None = None


async def embed_texts(
    texts: list[str], *, api_key: str | None = None
) -> Result[list[list[float]], EmbedError]:
    """Embed raw strings, returning one vector per input (Gemini → local fallback).

    Local by default (reliable, no rate limits). Gemini is opt-in via KAIRO_GEMINI_EMBED=1.
    The backend is sticky: once a run commits to local we never re-try Gemini, and once
    Gemini has produced vectors we never silently drop to local (which would mix
    dimensions) — a later Gemini failure surfaces as an error instead.
    """
    global _backend
    if not texts:
        return Ok([])

    key = api_key or os.environ.get("GEMINI_API_KEY")
    use_gemini = bool(key) and os.environ.get("KAIRO_GEMINI_EMBED") == "1"

    if use_gemini and _backend != "local":
        try:
            vectors = await asyncio.to_thread(_embed_gemini, texts, key)
            _backend = "gemini"
            log.info("Embedded %d texts via Gemini (%s)", len(texts), GEMINI_MODEL)
            return Ok(vectors)
        except Exception as exc:  # quota, auth, network
            if _backend == "gemini":
                return Err(EmbedError(reason=f"Gemini embedding failed mid-run: {exc}"))
            log.warning(
                "Gemini embedding failed (%s); using local %s for this run", exc, LOCAL_MODEL
            )

    try:
        vectors = await asyncio.to_thread(_embed_local, texts)
        _backend = "local"
        log.info("Embedded %d texts via local %s", len(texts), LOCAL_MODEL)
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
