from __future__ import annotations

from datetime import UTC, datetime
from io import StringIO

import pytest
from rich.console import Console

from ingestion.types import SynthesisEntryPoint, SynthesisModule, SynthesisResult
from output import renderer

# ANSI color codes rich emits for the basic colours (force_terminal=True).
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"


def _result(
    *,
    complexity: int = 5,
    n_modules: int = 1,
    circular_risks: list[str] | None = None,
) -> SynthesisResult:
    return SynthesisResult(
        repo_id="a" * 64,
        architecture_summary="A small layered service.",
        modules=[
            SynthesisModule(name=f"mod{i}", path=f"pkg/mod{i}.py", responsibility="does things")
            for i in range(n_modules)
        ],
        key_dependencies=["requests"],
        circular_risks=circular_risks if circular_risks is not None else [],
        entry_points=[SynthesisEntryPoint(name="main", file="cli/main.py", description="entry")],
        contributor_quickstart=["clone", "install", "test"],
        complexity_score=complexity,
        generated_at=datetime.now(UTC),
    )


def _render_to_string(result, **kwargs) -> str:
    buf = StringIO()
    # Patch the module console so we capture exactly what render() prints, with colour.
    original = renderer.console
    renderer.console = Console(file=buf, force_terminal=True, width=100)
    try:
        renderer.render(result, repo_name="myrepo", **kwargs)
    finally:
        renderer.console = original
    return buf.getvalue()


def test_circular_risks_render_red_panel() -> None:
    out = _render_to_string(_result(circular_risks=["auth -> db -> auth"]))
    assert "Circular Dependencies" in out
    assert RED in out  # the panel is red-styled


def test_no_circular_risks_no_panel() -> None:
    out = _render_to_string(_result(circular_risks=[]))
    assert "Circular Dependencies" not in out


def test_more_than_ten_modules_truncates() -> None:
    out = _render_to_string(_result(n_modules=14))
    assert "...and 4 more" in out


def test_complexity_color_green() -> None:
    out = _render_to_string(_result(complexity=3))
    assert GREEN in out


def test_complexity_color_yellow() -> None:
    out = _render_to_string(_result(complexity=6))
    assert YELLOW in out


def test_complexity_color_red() -> None:
    out = _render_to_string(_result(complexity=10))
    assert RED in out


def test_repo_name_and_summary_present() -> None:
    out = _render_to_string(_result())
    assert "myrepo" in out
    assert "small layered service" in out


@pytest.mark.parametrize("score,expected", [(1, GREEN), (4, YELLOW), (7, RED)])
def test_complexity_boundaries(score, expected) -> None:
    out = _render_to_string(_result(complexity=score))
    assert expected in out
