"""KairoRM CLI — the single entry point that wires the whole pipeline together.

`kairo map <source>` runs, end to end: fetch → filter → parse → index → agents →
synthesize → render/export → Q&A server. Every layer returns a `Result`; the first
error is printed in red and the process exits 1 — a user never sees a traceback.

This module is also where `RankResult.cycles` is threaded into the orchestrator (the
permitted wiring fix): the full pipeline is assembled here, so this is the one place
the cycles actually flow through to the dependency agent.
"""

from __future__ import annotations

import os
import sys

# Quiet grpc's noisy fork/poll warnings before any grpc-backed library loads.
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"
# epoll1 is a Linux-only poll engine; forcing it on macOS makes grpc fail to
# initialize ("No event engine could be initialized from epoll1") and hangs the
# pipeline. Only set it where it's valid.
if sys.platform.startswith("linux"):
    os.environ.setdefault("GRPC_POLL_STRATEGY", "epoll1")

import asyncio  # noqa: E402
import hashlib  # noqa: E402
import threading  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import click  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.progress import Progress, SpinnerColumn, TextColumn  # noqa: E402

from agents import orchestrator  # noqa: E402
from cli.banner import print_banner  # noqa: E402
from indexing import embeddings, vectorstore  # noqa: E402
from ingestion import detector, fetcher  # noqa: E402
from ingestion import filter as file_filter  # noqa: E402
from output import exporter, qa_server, renderer  # noqa: E402
from parsing import ast_parser, chunker, ranker  # noqa: E402
from synthesis import compressor, synthesizer  # noqa: E402

console = Console(stderr=False)
err_console = Console(stderr=True)

OUTPUT_DIR = Path("./kairomap-output")
DB_PATH = OUTPUT_DIR / ".chroma"
CACHE_DIR = OUTPUT_DIR / ".cache"


def _repo_name(source: str) -> str:
    cleaned = source.rstrip("/").removesuffix(".git")
    name = cleaned.split("/")[-1] or cleaned
    return name or "repo"


def _repo_id(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _describe(error: object) -> str:
    return str(getattr(error, "reason", error))


class _PipelineExit(SystemExit):
    """Carries exit code 1 without surfacing a traceback to the user."""


def _fail(message: str) -> None:
    err_console.print(f"[bold red]✗ {message}[/]")
    raise _PipelineExit(1)


async def main(source: str) -> None:
    """Run the full KairoRM pipeline against `source` (GitHub URL, zip, or local dir)."""
    print_banner()
    repo_name = _repo_name(source)
    repo_id = _repo_id(source)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # One Progress instance, one task per stage. Each task has total=1 and is marked
    # completed when its stage finishes, so the spinner stops (finished_text shows a
    # ✓) instead of animating forever — which is what made earlier stages appear to
    # repeat. No task is ever re-added.
    progress_columns = (
        SpinnerColumn(finished_text="[green]✓[/]"),
        TextColumn("[progress.description]{task.description}"),
    )
    with Progress(*progress_columns, console=console, transient=False) as progress:
        # 1 — Fetch
        task = progress.add_task("Fetching repo…", total=1)
        fetch_result = await fetcher.fetch(source, cache_dir=CACHE_DIR)
        if not fetch_result.is_ok():
            _fail(f"Fetch failed: {_describe(fetch_result.error)}")
        repo = fetch_result.unwrap()
        progress.update(task, completed=1, description=f"Fetched [cyan]{repo_name}[/]")

        # 2 — Filter + detect language
        task = progress.add_task("Filtering files…", total=1)
        raw_files = [rf async for rf in file_filter.walk(repo)]
        source_files = [detector.detect(rf) for rf in raw_files]
        progress.update(
            task, completed=1, description=f"Filtered [cyan]{len(source_files)}[/] files"
        )

        # 3 — Parse + chunk + rank
        task = progress.add_task("Parsing AST…", total=1)
        units = []
        for sf in source_files:
            parsed = ast_parser.parse(sf)
            if parsed.is_ok():
                units.extend(parsed.unwrap())
        chunks = chunker.chunk(units)
        rank_result = ranker.rank(chunks)
        progress.update(task, completed=1, description=f"Parsed [cyan]{len(chunks)}[/] chunks")

        # 4 — Index (embed + persist)
        task = progress.add_task("Indexing…", total=1)
        embed_result = await embeddings.embed(rank_result.chunks)
        if not embed_result.is_ok():
            _fail(f"Embedding failed: {_describe(embed_result.error)}")
        embedded = embed_result.unwrap()
        store_result = await vectorstore.store(embedded, repo_id=repo_id, db_path=DB_PATH)
        if not store_result.is_ok():
            _fail(f"Indexing failed: {_describe(store_result.error)}")
        progress.update(task, completed=1, description=f"Indexed [cyan]{len(embedded)}[/] chunks")

        # 5 — Agents (cycles threaded through to the dependency agent)
        task = progress.add_task("Running agents…", total=1)
        agents_result = await orchestrator.run_all(
            rank_result.chunks,
            repo_id=repo_id,
            db_path=DB_PATH,
            cycles=rank_result.cycles,
        )
        if not agents_result.is_ok():
            _fail(f"Agents failed: {_describe(agents_result.error)}")
        outputs = agents_result.unwrap()
        progress.update(task, completed=1, description="Agents analysed the codebase")

        # 6 — Synthesize + compress
        task = progress.add_task("Synthesizing…", total=1)
        synth_result = await synthesizer.synthesize(
            outputs, rank_result.chunks, repo_id=repo_id, repo_name=repo_name
        )
        if not synth_result.is_ok():
            _fail(f"Synthesis failed: {_describe(synth_result.error)}")
        result = synth_result.unwrap()
        compressed = compressor.compress(result)
        progress.update(task, completed=1, description="Synthesised the analysis")

    # Stats for the report header: file count, indexed chunks, language breakdown.
    lang_counts = Counter(sf.language for sf in source_files if sf.language != "unknown")
    stats = {
        "files": len(source_files),
        "chunks": len(embedded),
        "languages": dict(lang_counts.most_common()),
    }

    # 7 — Render the report (printed after the live region closes, so tables are clean)
    renderer.render(result, repo_name=repo_name, compressed=compressed)

    # 8 — Export to disk
    export_result = exporter.export(result, compressed, output_dir=OUTPUT_DIR, repo_name=repo_name)
    if not export_result.is_ok():
        _fail(f"Export failed: {_describe(export_result.error)}")
    manifest = export_result.unwrap()
    console.print(f"[green]✓ Exported to[/] {manifest.output_dir}")

    # 8 — Launch the Q&A server.
    # uvicorn.run() calls asyncio.run() internally, which explodes if invoked from
    # within this already-running event loop. Run it in a separate (daemon) thread —
    # that thread has no running loop, so uvicorn can spin up its own. We join the
    # thread so the process stays alive serving (the fake `start` in tests returns
    # immediately, so this doesn't block there).
    server_thread = threading.Thread(
        target=qa_server.start,
        kwargs={
            "compressed": compressed,
            "repo_name": repo_name,
            "repo_id": repo_id,
            "db_path": DB_PATH,
            "result": result,
            "stats": stats,
        },
        daemon=True,
    )
    server_thread.start()
    server_thread.join()


@click.group()
def cli() -> None:
    """KairoRM — code intelligence engine."""


@cli.command(name="map")
@click.argument("source")
def map_cmd(source: str) -> None:
    """Analyse a repo (GitHub URL, .zip, or local path) and launch its Q&A server."""
    # Load keys from a .env in the current directory before anything else, so users
    # don't have to export them every session. Explicit cwd path: the default
    # load_dotenv() searches from this module's location (site-packages when
    # installed), not where the user actually ran `kairo`.
    load_dotenv(dotenv_path=Path.cwd() / ".env")
    if not os.environ.get("GROQ_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        err_console.print(
            "[bold red]⚠ No LLM key found. Set GROQ_API_KEY or GEMINI_API_KEY in your "
            "environment or create a .env file in this directory.[/]"
        )
        raise SystemExit(1)

    try:
        asyncio.run(main(source))
    except _PipelineExit:
        raise  # already reported in red; just exit 1
    except Exception as exc:  # never leak a traceback to the user
        err_console.print(f"[bold red]✗ Unexpected error: {exc}[/]")
        raise SystemExit(1) from None
