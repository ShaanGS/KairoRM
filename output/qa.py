"""Q&A over an analysed repo — retrieval + a streamed answer, no UI.

Pure logic consumed by the interactive TUI: retrieve the most relevant code chunks for
a question, prepend the compressed synthesis context as the system-of-record, and stream
the LLM's answer (Groq → Gemini via the agents' shared streaming dispatch). Local tool,
so no auth.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from agents.base import _stream_text
from indexing import retriever
from ingestion.types import CompressedContext, RankedChunk

RETRIEVE_K = 10

_QA_SYSTEM_PROMPT = (
    "You are KairoRM, a code intelligence assistant answering questions about a specific "
    "repository. You are given a compressed analysis of the repo and the most relevant "
    "code chunks. Answer the user's question concisely and accurately, grounded in the "
    "provided material. Use markdown. If the material does not contain the answer, say so "
    "plainly rather than guessing."
)


def _format_chunks(chunks: list[RankedChunk]) -> str:
    if not chunks:
        return "(no specific code chunks retrieved)"
    parts = []
    for rc in chunks:
        c = rc.chunk
        parts.append(f"### {c.file_path} :: {c.unit_type} {c.name}\n{c.content}")
    return "\n\n".join(parts)


async def retrieve_context(
    question: str, *, repo_id: str | None, db_path: Path | None, k: int = RETRIEVE_K
) -> list[RankedChunk]:
    """Return the most relevant chunks for `question`, or [] if retrieval isn't available."""
    if not (repo_id and db_path is not None):
        return []
    try:
        result = await retriever.retrieve(question, repo_id=repo_id, db_path=db_path, k=k)
    except Exception:
        return []
    return result.unwrap() if result.is_ok() else []


async def answer(
    question: str,
    *,
    compressed: CompressedContext,
    repo_id: str | None = None,
    db_path: Path | None = None,
) -> tuple[list[RankedChunk], AsyncIterator[str]]:
    """Retrieve grounding chunks and return them with a token stream of the answer."""
    chunks = await retrieve_context(question, repo_id=repo_id, db_path=db_path)
    prompt = (
        f"## Repository analysis\n{compressed.content}\n\n"
        f"## Relevant code\n{_format_chunks(chunks)}\n\n"
        f"## Question\n{question}"
    )
    return chunks, _stream_text("qa", _QA_SYSTEM_PROMPT, prompt)
