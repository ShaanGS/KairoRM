"""FastAPI Q&A server over a synthesized repo.

A single `POST /ask` endpoint answers questions about the analysed codebase: it
retrieves the most relevant chunks from the vector store, prepends the compressed
synthesis context as a system-of-record, and asks the LLM (Groq → Gemini via the
agents' shared `_complete_text`). Local tool, so no auth.

`create_app` builds the ASGI app (used directly by tests via TestClient); `start`
wraps it in uvicorn for real use.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from rich.console import Console

from agents.base import _complete_text
from indexing import retriever
from ingestion.types import CompressedContext, RankedChunk, SynthesisResult
from output.web_page import build_report_html

console = Console(stderr=False)

RETRIEVE_K = 10

_QA_SYSTEM_PROMPT = (
    "You are KairoRM, a code intelligence assistant answering questions about a "
    "specific repository. You are given a compressed analysis of the repo and the most "
    "relevant code chunks. Answer the user's question concisely and accurately, "
    "grounded in the provided material. If the material does not contain the answer, "
    "say so plainly."
)


class AskRequest(BaseModel):
    # min_length=1 makes an empty question a 422 validation error.
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str
    chunks_used: int


def _format_chunks(chunks: list[RankedChunk]) -> str:
    if not chunks:
        return "(no specific code chunks retrieved)"
    parts = []
    for rc in chunks:
        c = rc.chunk
        parts.append(f"### {c.file_path} :: {c.unit_type} {c.name}\n{c.content}")
    return "\n\n".join(parts)


def create_app(
    compressed: CompressedContext,
    *,
    repo_name: str,
    repo_id: str | None = None,
    db_path: Path | None = None,
    result: SynthesisResult | None = None,
    stats: dict | None = None,
) -> FastAPI:
    app = FastAPI(title=f"KairoRM Q&A — {repo_name}")

    if result is not None:
        page = build_report_html(result, repo_name=repo_name, stats=stats or {})

        @app.get("/", response_class=HTMLResponse)
        async def report() -> str:
            return page

    @app.post("/ask", response_model=AskResponse)
    async def ask(request: AskRequest) -> AskResponse:
        chunks: list[RankedChunk] = []
        if repo_id and db_path is not None:
            result = await retriever.retrieve(
                request.question, repo_id=repo_id, db_path=db_path, k=RETRIEVE_K
            )
            if result.is_ok():
                chunks = result.unwrap()

        prompt = (
            f"## Repository analysis\n{compressed.content}\n\n"
            f"## Relevant code\n{_format_chunks(chunks)}\n\n"
            f"## Question\n{request.question}"
        )
        # json_mode=False: the answer is prose, not a JSON object (Groq rejects
        # json_object response_format unless the prompt mentions "json").
        llm = await _complete_text("qa", _QA_SYSTEM_PROMPT, prompt, json_mode=False)
        answer = llm.unwrap() if llm.is_ok() else f"Error answering question: {llm.error.reason}"
        return AskResponse(answer=answer, chunks_used=len(chunks))

    return app


def start(
    compressed: CompressedContext,
    *,
    repo_name: str,
    repo_id: str | None = None,
    db_path: Path | None = None,
    result: SynthesisResult | None = None,
    stats: dict | None = None,
    port: int = 8000,
) -> None:
    """Build the app and serve it with uvicorn (blocks until interrupted)."""
    import uvicorn

    app = create_app(
        compressed,
        repo_name=repo_name,
        repo_id=repo_id,
        db_path=db_path,
        result=result,
        stats=stats,
    )
    console.print(f"[bold cyan]KairoRM report ready at[/] http://localhost:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
