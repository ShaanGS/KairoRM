from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from cli import main as cli_main
from ingestion.types import (
    AgentOutputs,
    Chunk,
    CodeUnit,
    CompressedContext,
    EmbeddedChunk,
    Err,
    ExportManifest,
    FetchedRepo,
    InvalidSourceError,
    Ok,
    RankedChunk,
    RankResult,
    RawFile,
    SourceFile,
    SynthesisResult,
)


def _fixtures(tmp_path: Path) -> dict:
    repo = FetchedRepo(root=tmp_path, source_url="s", commit_sha=None, fetched_at=datetime.now(UTC))
    raw = RawFile(
        path=tmp_path / "a.py",
        rel_path=Path("a.py"),
        size_bytes=10,
        sha256="0" * 64,
        oversized=False,
    )
    sf = SourceFile(raw=raw, language="python", parser_name="python")
    unit = CodeUnit(
        file_path=tmp_path / "a.py",
        language="python",
        unit_type="function",
        name="a",
        start_line=1,
        end_line=2,
        raw_source="def a(): pass",
        imports=(),
        calls=(),
        parent=None,
    )
    chunk = Chunk(
        chunk_id="id_a",
        file_path=tmp_path / "a.py",
        language="python",
        unit_type="function",
        name="a",
        start_line=1,
        end_line=2,
        content="def a(): pass",
        token_count=4,
        imports=(),
        calls=(),
        context_header="# python | function a",
    )
    rchunk = RankedChunk(chunk=chunk, importance_score=0.5)
    rank_result = RankResult(chunks=[rchunk], cycles=[("id_a", "id_b")])
    embedded = EmbeddedChunk(ranked=rchunk, embedding=[0.1, 0.2])
    outputs = AgentOutputs(modules={"modules": []}, arch={}, deps={}, contributor={})
    synth = SynthesisResult(
        repo_id="r" * 64,
        architecture_summary="x",
        modules=[],
        key_dependencies=[],
        circular_risks=[],
        entry_points=[],
        contributor_quickstart=[],
        complexity_score=5,
        generated_at=datetime.now(UTC),
    )
    compressed = CompressedContext(content="ctx", token_count=1, truncated=False)
    manifest = ExportManifest(output_dir=tmp_path / "out", files=[tmp_path / "a.md"], repo_name="r")
    return locals()


@contextlib.contextmanager
def _patch_pipeline(
    monkeypatch, fx: dict, order: list[str], *, fetch_error=None, capture=None, interactive=True
):
    async def fetch(*a, **k):
        order.append("fetch")
        return fetch_error if fetch_error is not None else Ok(fx["repo"])

    async def walk(repo):
        order.append("walk")
        yield fx["raw"]

    def detect(rf):
        order.append("detect")
        return fx["sf"]

    def parse(sf):
        order.append("parse")
        return Ok([fx["unit"]])

    def chunk(units):
        order.append("chunk")
        return [fx["chunk"]]

    def rank(chunks):
        order.append("rank")
        return fx["rank_result"]

    async def embed(chunks):
        order.append("embed")
        return Ok([fx["embedded"]])

    async def store(chunks, **k):
        order.append("store")
        return Ok(None)

    async def run_all(chunks, **k):
        order.append("run_all")
        if capture is not None:
            capture["run_all_kwargs"] = k
        return Ok(fx["outputs"])

    async def synthesize(outputs, top_chunks, **k):
        order.append("synthesize")
        return Ok(fx["synth"])

    def compress(result, **k):
        order.append("compress")
        return fx["compressed"]

    def render(result, **k):
        order.append("render")

    def export(result, compressed, **k):
        order.append("export")
        return Ok(fx["manifest"])

    monkeypatch.setattr(cli_main.fetcher, "fetch", fetch)
    monkeypatch.setattr(cli_main.file_filter, "walk", walk)
    monkeypatch.setattr(cli_main.detector, "detect", detect)
    monkeypatch.setattr(cli_main.ast_parser, "parse", parse)
    monkeypatch.setattr(cli_main.chunker, "chunk", chunk)
    monkeypatch.setattr(cli_main.ranker, "rank", rank)
    monkeypatch.setattr(cli_main.embeddings, "embed", embed)
    monkeypatch.setattr(cli_main.vectorstore, "store", store)
    monkeypatch.setattr(cli_main.orchestrator, "run_all", run_all)
    monkeypatch.setattr(cli_main.synthesizer, "synthesize", synthesize)
    monkeypatch.setattr(cli_main.compressor, "compress", compress)
    monkeypatch.setattr(cli_main.renderer, "render", render)
    monkeypatch.setattr(cli_main.exporter, "export", export)
    monkeypatch.setattr(cli_main, "_interactive", lambda: interactive)
    yield


@pytest.mark.asyncio
async def test_full_pipeline_returns_tui_kwargs(monkeypatch, tmp_path: Path) -> None:
    fx = _fixtures(tmp_path)
    order: list[str] = []
    with _patch_pipeline(monkeypatch, fx, order):
        tui_kwargs, out_dir = await cli_main.main("https://github.com/psf/requests")
    # Interactive: main returns the console kwargs (the caller launches the TUI).
    assert tui_kwargs is not None
    assert out_dir is not None


@pytest.mark.asyncio
async def test_all_stages_called_in_order(monkeypatch, tmp_path: Path) -> None:
    fx = _fixtures(tmp_path)
    order: list[str] = []
    capture: dict = {}
    with _patch_pipeline(monkeypatch, fx, order, capture=capture):
        tui_kwargs, _ = await cli_main.main("https://github.com/psf/requests")

    # main() runs the pipeline through export; the TUI is launched by the caller, not here.
    expected = [
        "fetch",
        "walk",
        "detect",
        "parse",
        "chunk",
        "rank",
        "embed",
        "store",
        "run_all",
        "synthesize",
        "compress",
        "export",
    ]
    assert order == expected
    # The permitted wiring fix: rank_result.cycles flow into the orchestrator.
    assert capture["run_all_kwargs"]["cycles"] == [("id_a", "id_b")]
    # The returned kwargs carry the full result + stats the console needs.
    assert tui_kwargs["repo_id"] and "files" in tui_kwargs["stats"]


@pytest.mark.asyncio
async def test_non_tty_falls_back_to_static_render(monkeypatch, tmp_path: Path) -> None:
    fx = _fixtures(tmp_path)
    order: list[str] = []
    with _patch_pipeline(monkeypatch, fx, order, interactive=False):
        tui_kwargs, _ = await cli_main.main("https://github.com/psf/requests")
    # No terminal → static render, and no TUI kwargs returned.
    assert tui_kwargs is None
    assert "render" in order


def test_map_cmd_launches_tui_at_top_level(monkeypatch, tmp_path: Path) -> None:
    # The actual Bug-2 path: map_cmd must call KairoConsole(...).run() after the pipeline.
    from click.testing import CliRunner

    launched: dict = {}

    class _FakeConsole:
        def __init__(self, **kwargs):
            launched["kwargs"] = kwargs

        def run(self):
            launched["ran"] = True

    async def fake_main(source):
        return ({"repo_name": "x", "stats": {}}, tmp_path / "out")

    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setattr(cli_main, "load_dotenv", lambda **k: None)  # don't read a real .env
    monkeypatch.setattr(cli_main, "main", fake_main)
    monkeypatch.setattr(cli_main.tui, "KairoConsole", _FakeConsole)
    monkeypatch.setattr(cli_main.time, "sleep", lambda _s: None)

    result = CliRunner().invoke(cli_main.cli, ["map", "https://github.com/x/y"])
    assert result.exit_code == 0, result.output
    assert launched.get("ran") is True  # the TUI was launched via top-level .run()


@pytest.mark.asyncio
async def test_empty_repo_exits_gracefully(monkeypatch, tmp_path: Path) -> None:
    fx = _fixtures(tmp_path)
    order: list[str] = []
    buf = StringIO()
    monkeypatch.setattr(cli_main, "err_console", Console(file=buf, force_terminal=False))
    with _patch_pipeline(monkeypatch, fx, order):
        monkeypatch.setattr(cli_main.chunker, "chunk", lambda units: [])  # nothing parseable
        with pytest.raises(SystemExit) as exc_info:
            await cli_main.main("https://github.com/x/y")
    assert exc_info.value.code == 1
    assert "No source code" in buf.getvalue()
    # Stopped before indexing/agents — never burned an LLM call on an empty repo.
    assert "run_all" not in order


@pytest.mark.asyncio
async def test_fetch_error_exits_code_1_without_traceback(monkeypatch, tmp_path: Path) -> None:
    fx = _fixtures(tmp_path)
    order: list[str] = []
    err = Err(InvalidSourceError(source="bad", reason="nope"))

    buf = StringIO()
    monkeypatch.setattr(cli_main, "err_console", Console(file=buf, force_terminal=False))

    with _patch_pipeline(monkeypatch, fx, order, fetch_error=err):
        with pytest.raises(SystemExit) as exc_info:
            await cli_main.main("not-a-real-source")

    assert exc_info.value.code == 1
    out = buf.getvalue()
    assert "Fetch failed" in out
    assert "nope" in out
    # The pipeline stopped at fetch — nothing downstream ran.
    assert order == ["fetch"]
