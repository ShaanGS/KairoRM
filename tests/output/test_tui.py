from __future__ import annotations

from datetime import UTC, datetime

import pytest
from textual.widgets import Input

from ingestion.types import CompressedContext, SynthesisEntryPoint, SynthesisModule, SynthesisResult
from output import qa
from output.tui import KairoConsole, build_map_markdown


def _result() -> SynthesisResult:
    return SynthesisResult(
        repo_id="r" * 64,
        architecture_summary="A layered pipeline. Each stage feeds the next.",
        modules=[
            SynthesisModule(name="ingestion", path="ingestion", responsibility="Fetches code.")
        ],
        key_dependencies=["click", "rich"],
        circular_risks=["a.py -> b.py"],
        entry_points=[
            SynthesisEntryPoint(name="main", file="cli/main.py", description="Entry point.")
        ],
        contributor_quickstart=["Clone the repo", "Run the tests"],
        complexity_score=6,
        generated_at=datetime(2026, 6, 15, tzinfo=UTC),
    )


_STATS = {"files": 12, "chunks": 30, "languages": {"python": 12}}


def test_build_map_markdown_includes_everything() -> None:
    md = build_map_markdown(_result(), _STATS)
    assert "A layered pipeline." in md
    assert "ingestion" in md and "Fetches code." in md
    assert "cli/main.py" in md  # entry point
    assert "click" in md  # dependency
    assert "a.py -> b.py" in md  # circular risk
    assert "Clone the repo" in md  # contributor step
    assert "6/10" in md  # complexity


def _app() -> KairoConsole:
    return KairoConsole(
        result=_result(),
        stats=_STATS,
        compressed=CompressedContext(content="ctx", token_count=1, truncated=False),
        repo_id="r" * 64,
        db_path=None,
        repo_name="demo",
    )


@pytest.mark.asyncio
async def test_tui_composes_core_widgets() -> None:
    app = _app()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("#header")
        assert app.query_one("#map")
        assert app.query_one("#chat")
        assert app.query_one("#prompt", Input)


@pytest.mark.asyncio
async def test_tui_answers_a_question_via_stream(monkeypatch) -> None:
    async def fake_stream():
        for tok in ("Hello ", "world"):
            yield tok

    async def fake_answer(question, **kwargs):
        assert question == "hi"
        return [], fake_stream()

    monkeypatch.setattr(qa, "answer", fake_answer)

    app = _app()
    async with app.run_test() as pilot:
        app.query_one("#prompt", Input).focus()
        await pilot.press("h", "i", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        # The user's question is echoed and the answer worker finished cleanly.
        user_msgs = [str(s.render()) for s in app.query("#chat .user-msg")]
        assert any("hi" in t for t in user_msgs)
        assert app._busy is False
