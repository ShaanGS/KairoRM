"""Persistent vector storage backed by a local ChromaDB.

Each repository gets its own collection keyed by `repo_id`. Storing is idempotent:
if a populated collection already exists for this repo we skip re-indexing and let
callers `load` straight from disk — that's what makes repeated `kairo ask` calls
instant instead of re-embedding the whole codebase every time.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from indexing.embeddings import _silence_fd_stderr
from ingestion.types import Chunk, EmbeddedChunk, Err, Ok, RankedChunk, Result

log = logging.getLogger("kairo")


@dataclass(frozen=True, slots=True)
class StoreError:
    reason: str


def _collection_name(repo_id: str) -> str:
    # Chroma collection names are bounded; a sha256 prefix is unique enough per repo.
    return f"kairo_{repo_id[:56]}"


def _client(db_path: Path):  # noqa: ANN202 — chromadb.api.ClientAPI
    # chromadb's Rust core / default ONNX embedding function can print an HF-Hub
    # "unauthenticated requests" notice straight to fd 2 on first use; silence that fd
    # for the duration. We always supply our own vectors, so chroma's EF is never needed.
    with _silence_fd_stderr():
        import chromadb
        from chromadb.config import Settings

        db_path.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(
            path=str(db_path), settings=Settings(anonymized_telemetry=False)
        )


def _metadata(ec: EmbeddedChunk) -> dict:
    c = ec.ranked.chunk
    return {
        "chunk_id": c.chunk_id,
        "file_path": str(c.file_path),
        "language": c.language,
        "unit_type": c.unit_type,
        "name": c.name,
        "importance_score": float(ec.ranked.importance_score),
        "token_count": int(c.token_count),
        "start_line": int(c.start_line),
        "end_line": int(c.end_line),
        "imports_json": json.dumps(list(c.imports)),
        "calls_json": json.dumps(list(c.calls)),
        "context_header": c.context_header,
    }


def _reconstruct(document: str, embedding, metadata: dict) -> EmbeddedChunk:  # noqa: ANN001
    chunk = Chunk(
        chunk_id=metadata["chunk_id"],
        file_path=Path(metadata["file_path"]),
        language=metadata["language"],
        unit_type=metadata["unit_type"],
        name=metadata["name"],
        start_line=int(metadata["start_line"]),
        end_line=int(metadata["end_line"]),
        content=document,
        token_count=int(metadata["token_count"]),
        imports=tuple(json.loads(metadata["imports_json"])),
        calls=tuple(json.loads(metadata["calls_json"])),
        context_header=metadata["context_header"],
    )
    ranked = RankedChunk(chunk=chunk, importance_score=float(metadata["importance_score"]))
    return EmbeddedChunk(ranked=ranked, embedding=[float(x) for x in embedding])


def _store_sync(
    chunks: list[EmbeddedChunk], repo_id: str, db_path: Path
) -> Result[None, StoreError]:
    try:
        with _silence_fd_stderr():
            client = _client(db_path)
            name = _collection_name(repo_id)
            already_indexed = False
            try:
                existing = client.get_collection(name, embedding_function=None)
                already_indexed = existing.count() > 0
            except Exception:
                pass  # collection doesn't exist yet — create it below
            if not already_indexed:
                # embedding_function=None: we always supply our own vectors, so chroma
                # never needs (and never downloads) its default ONNX model from HF.
                collection = client.get_or_create_collection(name, embedding_function=None)
                collection.add(
                    ids=[ec.ranked.chunk.chunk_id for ec in chunks],
                    embeddings=[ec.embedding for ec in chunks],
                    documents=[ec.ranked.chunk.content for ec in chunks],
                    metadatas=[_metadata(ec) for ec in chunks],
                )
        if already_indexed:
            log.info("Collection for repo %s… already indexed; skipping re-index", repo_id[:12])
        else:
            log.info("Indexed %d chunks for repo %s…", len(chunks), repo_id[:12])
        return Ok(None)
    except Exception as exc:
        return Err(StoreError(reason=str(exc)))


def _load_sync(repo_id: str, db_path: Path) -> Result[list[EmbeddedChunk], StoreError]:
    try:
        with _silence_fd_stderr():
            client = _client(db_path)
            name = _collection_name(repo_id)
            try:
                collection = client.get_collection(name, embedding_function=None)
            except Exception as exc:
                return Err(StoreError(reason=f"no collection for repo {repo_id[:12]}…: {exc}"))
            data = collection.get(include=["embeddings", "documents", "metadatas"])
        ids = data.get("ids") or []
        documents = data.get("documents") or []
        embeddings = data.get("embeddings")
        embeddings = embeddings if embeddings is not None else []
        metadatas = data.get("metadatas") or []

        out: list[EmbeddedChunk] = []
        for i in range(len(ids)):
            out.append(_reconstruct(documents[i], embeddings[i], metadatas[i]))
        return Ok(out)
    except Exception as exc:
        return Err(StoreError(reason=str(exc)))


async def store(
    chunks: list[EmbeddedChunk], *, repo_id: str, db_path: Path
) -> Result[None, StoreError]:
    """Persist embedded chunks for `repo_id`. Idempotent — re-stores are skipped."""
    if not chunks:
        return Ok(None)
    return await asyncio.to_thread(_store_sync, chunks, repo_id, db_path)


async def load(repo_id: str, db_path: Path) -> Result[list[EmbeddedChunk], StoreError]:
    """Load all embedded chunks previously stored for `repo_id`."""
    return await asyncio.to_thread(_load_sync, repo_id, db_path)
